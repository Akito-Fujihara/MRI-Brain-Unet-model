import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from utils.dataloader import DataTransform

class UNet_show_image():
  """
  function: This function can visualize the inference result of PSPnet model.
  """

  def __init__(self, img_path_list, anno_path_list, net, input_size, device):
    """
    input
    -------------------------------------
    img_path_list: List of image paths to infer
    anno_path_list: List of anno paths
    net: Pretrained UNet model
    input_size: The size of the image to enter into the model
    """
    self.img_path_list = img_path_list
    self.anno_path_list = anno_path_list
    self.net = net.to(device)
    self.device = device
    self.transform = DataTransform(input_size=input_size)

  def show_and_save_image(self, idx):
    """
    input
    -------------------------------------
    idx: Index number of the path
    -------------------------------------

    output
    -------------------------------------
    result: Inference result image
    -------------------------------------
    """
    image_file_path = self.img_path_list[idx]
    anno_file_path = self.anno_path_list[idx]

    img = Image.open(image_file_path)
    img_width, img_height = img.size

    anno_class_img = Image.open(anno_file_path) 
    anno_class_img = anno_class_img.convert('P')
    show_anno_img = anno_class_img
    p_palette = anno_class_img.getpalette()

    img, anno_class_img = self.transform("val", img, anno_class_img)

    self.net.eval()
    x = img.unsqueeze(0)
    x = x.to(self.device, dtype=torch.float)
    outputs = self.net(x)
    y = outputs
    device2 = torch.device('cpu')
    y = y.to(device2)
    y = y[0][0].detach().numpy()

    show_anno_img = show_anno_img.resize((img_width, img_height), Image.NEAREST)
    show_anno_img.putpalette(p_palette)
    anno_class_img = Image.fromarray(np.uint8(y), mode="P")
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)
    anno_class_img = anno_class_img.convert("L") 
    anno_class_img = anno_class_img.point(lambda x: 0 if x > 230 else 255)

    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')  
    show_anno_img = show_anno_img.convert('RGBA')

    for x in range(img_width):
        for y in range(img_height):
            pixel = anno_class_img.getpixel((x, y))
            r, g, b, a = pixel
            pixel_show = show_anno_img.getpixel((x, y))
            r_show, g_show, b_show, a_show = pixel_show

            if pixel_show[0] == 0 and pixel_show[1] == 0 and pixel_show[2] == 0:
                continue
            else:
                trans_img.putpixel((x, y), (0, 255, 255, 150))

            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            else:
                trans_img.putpixel((x, y), (255, 255, 0, 150))

    img = Image.open(image_file_path) 
    result = Image.alpha_composite(img.convert('RGBA'), trans_img)
    return cv2.cvtColor(np.asarray(result), cv2.COLOR_RGBA2BGRA)