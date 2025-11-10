'''
transform pytorch model to onnx model
'''
import torch
import torch.onnx
import onnx
import onnxruntime
import os

import sys
sys.path.append('../')
from model import select_model
from options import get_options

def convert2onnx(model, save_path, dummy_input, input_names, output_names, verbose=False):
    '''
    convert pytorch model to onnx model
    '''
    model.eval()
    torch.onnx.export(model, dummy_input, save_path, verbose=verbose,
                      input_names=input_names, output_names=output_names)
    
    print('convert to onnx model {} successfully!'.format(save_path))
    return

def load_onnx_model(onnx_path):
    '''
    load onnx model
    '''
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print('onnx model check successfully!')
    ort_session = onnxruntime.InferenceSession(onnx_path)
    return ort_session

def onnx_inference(ort_session, input_data):
    '''
    onnx inference
    '''
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def onnx_inference_with_io(ort_session, input_data, input_names, output_names):
    '''
    onnx inference with input and output names
    '''
    ort_inputs = {input_names[0]: input_data}
    ort_outs = ort_session.run(output_names, ort_inputs)
    return ort_outs

def onnx_inference_with_io2(ort_session, input_data, input_names, output_names):
    '''
    onnx inference with input and output names
    '''
    ort_inputs = {}
    for i in range(len(input_names)):
        ort_inputs[input_names[i]] = input_data[i]
    ort_outs = ort_session.run(output_names, ort_inputs)
    return ort_outs

def main():
    args = get_options()
    model = select_model(args)
    model.load_state_dict(torch.load(args.load_model_path))
    model_name = args.model_type
    save_root = '/home/haozheng/underwater-master/onnx_model/2/'
    load_model_path = ''
    folder_name = os.path.split(load_model_path)[-2]
    devices = ['cuda']
    batch_sizes = [1, 8]
    for device in devices:
        for batch_size in batch_sizes:
            model = model.to(device)
            dummy_input = torch.randn(batch_size, 3, 3, 3, device=device)
            input_names = ['input']
            output_names = ['output']
            save_path = os.path.join(save_root, folder_name, '{}_{}_{}.onnx'.format(model_name, device, batch_size))
            convert2onnx(model, save_path, dummy_input, input_names, output_names, verbose=False)

if __name__ == '__main__':
    main()
