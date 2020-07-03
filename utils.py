import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import deepmux
from consts import DEEPMUX_TOKEN



def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H * W)
    x_t = x.transpose(1, 2)
    return torch.bmm(x, x_t) / (C * H * W)


# Load image file
def load_image(path):
    img = cv2.imread(path)
    return img


# Show image
def show(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img / 255).clip(0, 1)

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.show()


def saveimg(img: np.ndarray, image_path='', container=False):
    img = img.clip(0, 255)
    if not container:
        cv2.imwrite(image_path, img)
        return True, None
    else:
        is_success, buffer = cv2.imencode(".jpg", img)
        return is_success, buffer


def itot(img, max_size=None):
    if max_size == None:
        itot_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H, W])) * x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    tensor = itot_t(img)
    tensor = tensor.unsqueeze(dim=0)
    return tensor


def ttoi(tensor):
    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()
    img = img.transpose(1, 2, 0)
    return img


def reg_models(model, styles):
    model_temp = model()
    names = []
    for style in styles[0:]:
        model_temp.load_state_dict(torch.load(style + ".pth"))
        model_name = style + '_pepilica_gan_project'
        model_nothing = deepmux.create_model(pytorch_model=model_temp,
                                             model_name=model_name,
                                             input_shape=[1, 3, 256, 256],
                                             output_shape=[1, 3, 256, 256],
                                             token=DEEPMUX_TOKEN)
        if not model_nothing:
            break
        names.append(model_name)
    return names
