import torchvision
import torch
import torch.optim
import os
import argparse
import model
import numpy as np
from PIL import Image


def image_haze_removel(input_image):
    hazy_image = (np.asarray(input_image) / 255.0)
    hazy_image = torch.from_numpy(hazy_image).float()
    hazy_image = hazy_image.permute(2, 0, 1)
    hazy_image = hazy_image.cuda().unsqueeze(0)
    with torch.no_grad():
        lfd_net = model.LFD_Net().cuda()
        lfd_net.load_state_dict(torch.load('outdoor.pth'))
        dehaze_image = lfd_net(hazy_image)
    return dehaze_image


def multiple_dehaze_test(test_directory, save_directory):
    for filename in os.listdir(test_directory):
        path = os.path.join(test_directory, filename)
        try:
            image = Image.open(path)
        except Exception:
            continue
        dehaze_image = image_haze_removel(image)
        torchvision.utils.save_image(dehaze_image, os.path.join(save_directory, filename))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-td", "--test_directory", required=True, help="path to test images directory")
    ap.add_argument("-sd", "--save_directory", required=True, help="path to save test results directory")
    args = vars(ap.parse_args())
    multiple_dehaze_test(args["test_directory"], args["save_directory"])
