import torch 
import torch.nn as nn 
import torch.nn.functional as F
# Inference helper 

def get_image(path: str, w : int = 256, h : int = 256) -> None:
     
    image = cv2.imread(path)
    image = cv2.resize(image, (w, h))
    t_image = tensor_from_rgb_image(image)
    return t_image


