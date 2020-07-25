import io 
import numpy as np 

from torch import nn 
import torch.utils.model_zoo as model_zoo 
import torch.onnx
import typing 
from typing import Typing import List, Dict, Union, Tuple 

model_path = 'results/epoch=16.ckpt'

class OnnxExport(nn.Module):

    def __init__(self, size :  Tuple, model : None, model_name : str ):

        self.size = size 
        self.model = model 

    def _export(self, path : str) -> None:
        
        x = torch.randn(1, size, requires_grad = True)
        torch_out = self.model(x)
        torch.onnx.export(self.model, 
                x, 
                model_name, 
                export_params = True, 
                opts_version  = 10, 
                do_constant_folding = True, 
                input_names = ['input'], 
                output_names = ['output'], 
                dynamic_axis = {'input', : {0:  'batch_size'}, 
                    'output' : {0 : 'batch_size'}})


def main():

    pass 



