import torch 
import torch.nn as nn 
import torch.nn.functional as F
import cv2
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

# Inference helper 

def get_image(path: str, w : int = 256, h : int = 256) -> None:
     
    image = cv2.imread(path)
    image = cv2.resize(image, (w, h))
    t_image = tensor_from_rgb_image(image)
    return t_image


