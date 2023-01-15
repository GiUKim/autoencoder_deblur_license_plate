import torch.onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
import os
import sys
import onnx
from onnx import shape_inference
import onnx.numpy_helper as numpy_helper
from numpy import asarray
from model import *
import numpy as np
import cv2
def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
    # Reference : https://gaussian37.github.io/python-basic-numpy-snippets/
    flag = False
    try :
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag

def Convert_ONNX(width, height, input_image_name=None):
    trans = transforms.ToTensor()
    if input_image_name is not None:
        if isColor:
            dummy_img = Image.open(input_image_name)
        else:
            dummy_img = Image.open(input_image_name).convert('L')
        dummy_img = dummy_img.resize((width, height))
        dummy_input = trans(dummy_img)

    else:
        if isColor:
            dummy_img = np.random.rand(height, width, 3) * 255
        else:
            dummy_img = np.random.rand(height, width, 1) * 255
        dummy_input = trans(dummy_img).float()
    print('dummy_input.shape:', dummy_input.shape)
    dummy_input = dummy_input.unsqueeze(0)

    dummy_output = model(dummy_input)
    print('dummy_output:', dummy_output)

    torch.onnx.export(model,
                      dummy_input,
                      "outcf.onnx",
                      verbose=True,
                      example_outputs=dummy_output,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['output']
                      )

    print("dummy_output:", dummy_output)
    print(" ")
    print("Model has been converted to ONNX")

    onnx_model = onnx.load('outcf.onnx')

    onnx_layers = dict()
    for layer in onnx_model.graph.initializer:
        onnx_layers[layer.name] = numpy_helper.to_array(layer)

    torch_layers = {}
    for layer_name, layer_value in model.named_modules():
        torch_layers[layer_name] = layer_value

    onnx_layers_set = set(onnx_layers.keys())
    torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
    filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

    difference_flag = False
    for layer_name in filtered_onnx_layers:
        onnx_layer_name = layer_name
        torch_layer_name = layer_name.replace(".weight", "")
        onnx_weight = onnx_layers[onnx_layer_name]
        torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
        flag = compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
        difference_flag = True if flag == True else False

    if difference_flag:
        print("update onnx weight from torch model.")
        for index, layer in enumerate(onnx_model.graph.initializer):
            layer_name = layer.name
            if layer_name in filtered_onnx_layers:
                onnx_layer_name = layer_name
                torch_layer_name = layer_name.replace(".weight", "")
                onnx_weight = onnx_layers[onnx_layer_name]
                torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
                copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
                onnx_model.graph.initializer[index].CopyFrom(copy_tensor)

        print("save updated onnx model.")
        onnx_new_path = os.path.dirname(os.path.abspath('outcf.onnx')) + os.sep + "updated_" + os.path.basename('outcf.onnx')
        onnx.save(onnx_model, onnx_new_path)

    if difference_flag:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load('new_outcf.onnx')), 'new_outcf.onnx')
    else:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load('outcf.onnx')), 'outcf.onnx')

if __name__ == "__main__":
    width, height = int(sys.argv[2]), int(sys.argv[3])
    color = sys.argv[4]
    if color == 'g':
        isColor = False
    else:
        isColor =  True
    dummy = np.random.rand(height, width, 3) * 255
    cv2.imwrite('./dummy.jpg', dummy)
    model = Net()
    path = './checkpoint/' + sys.argv[1]
    model.load_state_dict(torch.load(path))
    model.eval()
    Convert_ONNX(width, height, 'dummy.jpg')


