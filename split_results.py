import numpy as np
import argparse
import os

def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def get_time_file(name):
    time=(name[2].split('_')[1])[:-4]#.split('.')[0]
    time=np.int32(time)
    return time

def read_one_data(name,dir,save_dir):
    f=open(dir+'/'+name+".txt",'r')
    lines=f.readlines()
    
    id_list=[]
    for line in lines:
        data=line.split(',')
        id=data[5]#.replace('\n','')
        sonar=data[6].replace('\n','')
        id_sonar=[id,sonar]
        if id_sonar not in id_list:
            id_list.append(id_sonar)
        else:
            continue
    
    #print(id_list)
    for id_sonar in id_list:
        f1=open(save_dir+'/'+name+"_"+str(id_sonar[0])+"_"+str(id_sonar[1])+".txt",'w')
    #f1=open(save_dir+'/'+name+".txt",'w')
        data_re=[]
        for line in lines:
            line=line.replace('\n','')
            data=line.split(',')
            id_n=data[5]
            sonar_n=data[6].replace('\n','')
            if id_n==id_sonar[0] and sonar_n==id_sonar[1]:
                data_one=[data[2],data[3],data[4]]
                data_re.append(data_one)
        data_re.sort(key=get_time_file)
        for i in range(len(data_re)):
            f1.writelines(data_re[i][0]+','+data_re[i][1]+','+data_re[i][2]+'\n')
        f1.close()
        
def dir_convert(dir,save_dir):
    dir_create(save_dir)
    files=os.listdir(dir)
    for file in files:
        scenario=file.split('_')[0]
        print(scenario)
        #if scenario!="08071005" and scenario!="08071006":
        #    continue
        if file[0]==".":
            continue
        #print(file)
        name=file.split('.')[0]
        print(name,dir,save_dir)
        #print(name)
        read_one_data(name,dir,save_dir)
        
def statistic_single_file_results(file):
    classes=["moving","motionless","patting","struggling","drowning"]
    confuse_matrix=np.zeros((5,5))
    f=open(file,'r')
    lines=f.readlines()
    for line in lines:
        data=line.split(' ')
        pred_state=data[1]
        gt_state=data[2]
        index_cls_pred=0
        index_cls_gt=0
        for j in range(len(classes)):
            if classes[j]==pred_state:
                index_cls_pred=j
            if classes[j]==gt_state:
                index_cls_gt=j
        confuse_matrix[index_cls_pred][index_cls_gt]+=1
    return confuse_matrix

def statistic_all_files(dir_re):
    files=os.listdir(dir_re)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    for file in files:
        re_path=dir_re+"/"+file
        print(re_path)
        confuse_matrix=statistic_single_file_results(re_path)
        #print(file)
        print("pred/gt")
        print(confuse_matrix)
        print(" ")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="motion")
    parser.add_argument("--save_dir", type=str, required=True, help="split_results")
    parser.add_argument("--save_dir_all", type=str, required=True, help="moving")
    args = parser.parse_args()
    dir=args.dir
    save_dir=args.save_dir
    save_dir_all=args.save_dir_all
    dir_convert(dir,save_dir_all+"/"+save_dir)
 