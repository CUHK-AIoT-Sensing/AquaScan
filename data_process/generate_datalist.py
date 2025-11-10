#!/usr/bin/env python
# encoding: utf-8
"""
This script can read your required scenarios and generate data list.
"""
import argparse
import os

def walk_all_files(data_root, label_root, save_name):
    """
    Walk all files in data_root and label_root, and save the file names in save_name.
    Data and label files should have the same name.
    """
    with open(save_name, "w") as f:
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith(".png"):
                    data_file = os.path.join(root, file)
                    label_file = os.path.join(label_root, file.replace(".png", ".txt"))
                    f.write(os.path.abspath(data_file) + " " + os.path.abspath(label_file) + "\n")
                    
def walk_file_list(data_root,label_root,post_=".png"):
    data_label_list=[]
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(post_):
                data_file = os.path.join(root, file)
                label_file = os.path.join(label_root, file.replace(post_, ".txt"))
                data_label_list.append(os.path.abspath(data_file) + " " + os.path.abspath(label_file) + "\n")
    return data_label_list
    
def walk_all_list_skip(data_root,label_root,skip_root):
    data_label_list=[]
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".png"):
                data_file = os.path.join(root,file)
                label_file = os.path.join(label_root,file)
                skip_file = os.path.join(skip_root,file.replace(".png", ".txt"))
                data_label_list.append(os.path.abspath(data_file)+" "+os.path.abspath(label_file)+" "+os.path.abspath(skip_file)+ "\n")
    return data_label_list
    
def save_list_all(save_name,list):
    with open(save_name,"w") as f:
        for i in range(len(list)):
            for j in range(len(list[i])):
                f.write(list[i][j])

def main():
    parser = argparse.ArgumentParser(description="Generate data list")
    parser.add_argument("--data_root_path", type=str, help="root path of the processed data files")
    parser.add_argument("--label_root_path", type=str, help="root path of the processed label files")
    parser.add_argument("--save_name", type=str, help="datalist name (or include path)")

    args = parser.parse_args()
    data_root_path = args.data_root_path
    label_root_path = args.label_root_path
    save_name = args.save_name

    assert os.path.exists(data_root_path), f"{data_root_path} not found"
    assert os.path.exists(label_root_path), f"{label_root_path} not found"
    save_dir = os.path.abspath(save_name)
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    walk_all_files(data_root_path, label_root_path, save_name)


if __name__ == "__main__":
    main()
