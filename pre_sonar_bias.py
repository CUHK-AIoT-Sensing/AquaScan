import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from sklearn.cluster import DBSCAN
#from skimage import transform
import cmath
import math
import cv2
import os
import argparse
import copy
import time
import denoise_metric as dm

#this function is designed for errors caused by hardware. Sonar may shift to wrong angles when scanning.

def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def readline(line):
    ''' read one line of txt file
    Args:
        line: one line of txt file. Format: angle data0 data1 ... dataX
        By default, each line has 1 + 500 data.
    Returns:
        angle: angle of this line
        data: sonar data
    '''
    line = line.split()
    line = list(map(float, line))
    angle = line[0]
    data = line[1:]
    return angle, data
    
def read_txt(path, angle_range=400,bias=0):
    ''' read sonar data from txt file
    Args:
        path: path of txt file
        angle_range: range of angle (1 gradian = 0.9 degree), default: 400
    Returns:
        sonar_data: 2d array, shape: (angle_range, 500)
        start_angle: start angle of sonar data
        end_angle: end angle of sonar data
    '''
    #print(path)
    #s=path.split('/')[-1]
    path_seg=path.split('/')[-1]
    if '_' in path_seg:
        file_name=str(path_seg.split('_')[1])
    else:
        file_name=path_seg
    #print(file_name)
    file_order=file_name.split('.')[0]
    sonar_data = np.zeros((angle_range, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            if len(data) == 500:
                if np.int32(file_order)%2==1:
                    sonar_data[(int(angle)+bias)%400] = data
                else:
                    sonar_data[int(angle)] = data
    return sonar_data, int(start_angle), int(end_angle)

def read_default_label_noangle(file_path):
    '''
    Args:
        file_path: path of label file
    Returns:
        human_ids: list of human id
        states: list of state
        objs: list of object, each object is a list of [ymin, ymax, xmin, xmax]
    '''
    human_ids = []
    states = []
    objs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if '\n' in lines:
            lines.remove('\n')
        print(lines)
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[2])/2.0)
                ymin = int(float(arr[3])/2.0)
                xmax = int(float(arr[4])/2.0)
                ymax = int(float(arr[5])/2.0)
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[1])/2.0)
                ymin = int(float(arr[2])/2.0)
                xmax = int(float(arr[3])/2.0)
                ymax = int(float(arr[4])/2.0)

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def image_shift(data, bias):
    new_data=np.zeros_like(data)
    print(len(data))
    for i in range(len(data)):
        new_data[(i+bias)%400,:]=data[i,:]
    return new_data

def label_shift(obj_single,bias):
    print(obj_single)
    for i in range(len(obj_single)):
        obj_single[i][0]+=bias
        obj_single[i][1]+=bias
    return obj_single
    
def save_process_data(data,data_path,image_path,image_save=False):
    np.save(data_path,data)
    if image_save:
        cv2.imwrite(image_path,data)
        
def label_save(h,s,obj,save_path):
    f=open(save_path,"w")
    for i in range(len(obj)):
        record_single=str(h[i])+" "+str(s[i])+" "+str(obj[i][0])+" "+str(obj[i][1])+" "+str(obj[i][2])+" "+str(obj[i][3])+"\n"
        f.writelines(record_single)
    f.close()

def read_data_process(dirc,dirc_label,save_process_npy,save_img,save_label,label_scenario):
    scenario=os.listdir(dirc_label)
    dir_create(save_process_npy)
    dir_create(save_img)
    dir_create(save_label)
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    for i in range(len(scenario)): 
        if scenario[i] not in label_scenario: 
            continue   
        data_path=dirc+"/"+scenario[i]
        label_path=dirc_label+"/"+scenario[i]
        save_path_data=save_process_npy+"/"+scenario[i]
        save_img_data=save_img+"/"+scenario[i]
        save_label_data=save_label+"/"+scenario[i]
        sonars=os.listdir(data_path)
        dir_create(save_path_data)
        dir_create(save_img_data)
        dir_create(save_label_data)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            data_path_single=data_path+"/"+sonar
            label_path_single=label_path+"/"+sonar
            save_path_single=save_path_data+"/"+sonar
            save_img_single=save_img_data+"/"+sonar
            save_label_single=save_label_data+"/"+sonar
            dir_create(save_path_single)
            dir_create(save_label_single)
            dir_create(save_img_single)
            files = os.listdir(data_path_single)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            for file in files:
                npy_save_path=save_path_single+"/"+file[:-4]+".npy"
                img_save_path=save_img_single+"/"+file[:-4]+".png"
                img_save_path_raw=save_img_single+"/"+file[:-4]+"_raw.png"
                label_save_path=save_label_single+"/"+file
                data_path_single_file=data_path_single+'/'+file
                label_path_single_file=label_path_single+"/"+file
                sonar_data,_,_=read_txt(data_path_single_file)
                h,s,obj=read_default_label_noangle(label_path_single_file)
                index=np.int32(file[:-4].split("_")[1])
                print(index,obj)
                if (index+1)%2==0:
                    sonar_data=image_shift(sonar_data,12) #12 or 13
                    obj=label_shift(obj,12)
                save_process_data(sonar_data,npy_save_path,img_save_path,True)
                label_save(h,s,obj,label_save_path)

def pre_process_shift():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--label",type=str, required=True, help="label_path")
    args = parser.parse_args()
    label_sc=["111_raw"]
    save_img="./111_data"
    save_npy="./111_label"
    save_new_label="./pre_label_111"
    data_dir=args.data
    label_dir=args.label
    read_data_process(data_dir,label_dir,save_npy,save_img,save_new_label,label_sc)
    
    
if __name__=='__main__':
    pre_process_shift()
