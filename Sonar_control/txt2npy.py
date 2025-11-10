#  txt2npy convertor file
import os
import argparse
import numpy as np
def file_name(path):
    dict_file_name={}
    log_path=path
    count=0
    for root,dirs,files in os.walk(path):
        for file in files:
            dict_file_name.update({count:file})
            count+=1
    return dict_file_name

def txtreader2numpy(path,number_sample):
    f=open(path)
    lines=f.readlines()
    rows=len(lines)
    save_data=np.zeros((rows,number_sample+1))
    count=0
    for line in lines:
        line = line.strip().split(' ')
        for j in range(len(line)):
            save_data[count][j]=float(line[j])
        count+=1
    return save_data
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="read_data")
    parser.add_argument("--path",default="mode_0_only_sonar_rpi/",type=str,help="read dirct.")
    parser.add_argument("--save",default="post_process/",type=str,help="save dirct.")
    args=parser.parse_args()
    dict_file=file_name(args.path)
    for i in range(len(dict_file)):
        data=txtreader2numpy(args.path+dict_file[i],500)
        np.save(args.save+dict_file[i][:len(dict_file[i])-4]+".npy",data)
    
        
      
