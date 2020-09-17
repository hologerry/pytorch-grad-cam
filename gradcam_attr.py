import os
import argparse
import cv2
import numpy as np
import torch

from torch.autograd import Function
# from torchvision import models
from resnet import resnet18


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # print("FeatureExtractor module name", name)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers, num_attributes):
        self.model = model
        self.feature_module = feature_module
        self.num_attributes = num_attributes
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, attribute_idx):
        target_activations = []
        # print("---------------------- self.model ----------------------", self.model)
        # print("---------------------- self.model._modules -------------", self.model._modules)
        for name, module in self.model._modules.items():
            # print("Model Output module name", name)
            if "classifier" in name:
                break
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        y = []
        for i in range(self.num_attributes):
            classifier = self.model._modules['classifier' + str(i).zfill(2)]
            y.append(classifier(x))
        return target_activations, y[attribute_idx]


def preprocess_image(img):
    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    means = [0.5, 0.5, 0.5]
    stds = [0.5, 0.5, 0.5]
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask, file_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(f"{file_name}_cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, num_attrs):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names, num_attrs)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, attribute_idx, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda(), attribute_idx)
        else:
            features, output = self.extractor(input, attribute_idx)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, attribute_idx, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        output = output[attribute_idx]  # !!!
        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--all_attrs', type=list,
                        default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                                 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                                 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                                 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                                 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                                 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                                 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                                 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                                 'Wearing_Necktie', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3'])
    parser.add_argument('--select_attr', type=str, default='Male')
    parser.add_argument('--resume', default='checkpoints/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    if args.select_attr != '':
        args.resume = f'checkpoints_single_{args.select_attr}/model_best.pth.tar'

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.join(args.output_dir, args.image_path.split('/')[-1].split('.')[0])
    file_name = file_name + '_' + args.select_attr
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)
    print("=> creating attribute model resnet18")
    # model = resnet18(pretrained=True, num_attributes=len(args.all_attrs))  # here we hard code to resnet18
    model = resnet18(pretrained=True, num_attributes=1)  # here we hard code to resnet18
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded attribute checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

    model = model.module
    # print("----------------ResNet 18 model ------------")
    # print(model)
    # print("--------------------------------------------")
    grad_cam = GradCam(model=model, feature_module=model.layer4,
                       target_layer_names=["1"], use_cuda=args.use_cuda, num_attrs=1)

    img = cv2.imread(args.image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    img = np.float32(cv2.resize(img, (256, 256))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = 1
    # select_attr_idx = args.all_attrs.index('Eyeglasses')
    select_attr_idx = 0
    mask = grad_cam(input, select_attr_idx, target_index)

    show_cam_on_image(img, mask, file_name)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    gb = gb_model(input, attribute_idx=select_attr_idx, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{file_name}_gb.jpg', gb)
    cv2.imwrite(f'{file_name}_cam_gb.jpg', cam_gb)
