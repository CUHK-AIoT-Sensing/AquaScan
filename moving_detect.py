import numpy as np
import os
import time

def calculate_iou_on_small(obj1, obj2):
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    obj1_area = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2])
    obj2_area = (obj2[1] - obj2[0]) * (obj2[3] - obj2[2])
    iou = inter / min(obj1_area, obj2_area)
    return iou

def calculate_iou(obj1, obj2):
    '''pos: [ymin, ymax, xmin, xmax]
    '''
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    union = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2]) + (obj2[1] - obj2[0]) * (obj2[3] - obj2[2]) - inter
    return inter / union
    
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

def read_BBox_data(path):
    f=open(path)
    lines=f.readlines()
    lines=list(set(lines))
    pos=[]
    for line in lines:
        data=line.split(",")
        #print(data)
        if np.int32((np.int32(data[1])/(15.0/4.0))-1)<0:
            region_1=0
        else:
            region_1=np.int32((np.int32(data[1])/(15.0/4.0))-1)
        if np.int32((np.int32(data[0])/(12.0/5.0))-1)<0:
            region_3=0
        else:
            region_3=np.int32((np.int32(data[0])/(12.0/5.0))-1)
        pos_single=[region_1,np.int32((np.int32(data[3])/(15.0/4.0))+1),region_3,np.int32((np.int32(data[2])/(12.0/5.0))+1)]
        pos.append(pos_single)
    return pos

def read_yolo_label(file_path, W=500, H=400,bias=9):
    ''' Read one txt file in yolo format.
        human_id | state_str | x_center | y_center | width | height
    Args:
        file_path: path of label file
        W: width of sonar data, default 500 unit (20 m)
        H: height of sonar data, default 400 gradian
    Returns:
        obj_list: list of object, each object is a list of [ymin, ymax, xmin, xmax]
        states: list of state, each state is a string
    '''
    humans=[]
    obj_list = []
    states = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split()
            human_id = arr[0]
            state = arr[1]
            x_center = float(arr[2])
            y_center = float(arr[3])
            width = float(arr[4])
            height = float(arr[5])

            xmin = int((x_center - width / 2) * W)
            xmax = int((x_center + width / 2) * W)
            ymin = int((y_center - height / 2) * H)
            ymax = int((y_center + height / 2) * H)

            obj = [ymin, ymax, xmin, xmax]
            obj_list.append(obj)
            states.append(state)
            humans.append(human_id)
    return humans, states, obj_list

def read_default_label_raw(file_path):
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
        #print(lines)
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[4])) #4
                ymin = int(float(arr[2])) #2
                xmax = int(float(arr[5])) #5
                ymax = int(float(arr[3])) #3
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[3]))
                ymin = int(float(arr[1]))
                xmax = int(float(arr[4]))
                ymax = int(float(arr[2]))

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def calculate_iou(obj1, obj2):
    '''pos: [ymin, ymax, xmin, xmax]
    '''
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    union = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2]) + (obj2[1] - obj2[0]) * (obj2[3] - obj2[2]) - inter
    return inter / union

def generate_moving_BBox(human,states,label):
    merge_label=[]
    for j in range(len(label)):
        label_choose=j
        #label_new_single=[human[label_choose],states[label_choose],label[label_choose][0],label[label_choose][1],label[label_choose][2],label[label_choose][3]]
        #print(label_new_single)
        label_y=(label[label_choose][0]+label[label_choose][1])/2.0
        label_x=(label[label_choose][1]+label[label_choose][2])/2.0
        x=np.cos(np.deg2rad(label_y))*label_x*2.0
        y=np.sin(np.deg2rad(label_y))*label_x*2.0
        label_new_single=[human[label_choose],states[label_choose],label_y,label_x,y,x]
        merge_label.append(label_new_single)
    merge_label.sort(key=lambda x: np.int32(x[0]))
    localize=[]
    print(human)
    print(states)
    print(label)
    print(merge_label)
    print(" ")
    return merge_label

def read_moving(obj_dir,data_obj):
    print(obj_dir)
    count=0
    scenarios=os.listdir(obj_dir)
    loc_all=[]
    for scenario in scenarios:
        if scenario[0]==".":
            continue
        obj_dir_s=obj_dir+scenario
        data_obj_s=data_obj+scenario
        sonars=os.listdir(obj_dir_s)
        for sonar in sonars:
            loc=[]
            if sonar[0]==".":
                continue
            obj_dir_sonar=obj_dir_s+"/"+str(sonar)
            data_obj_sonar=data_obj_s+"/"+str(sonar)
            print(obj_dir_sonar)
            files=os.listdir(obj_dir_sonar)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            files.sort(key=lambda x:x[:-4].split("_")[1])    
            for file in files:
                obj_dir_file=obj_dir_sonar+"/"+file
                data_obj_file=data_obj_sonar+"/"+file
                h,s,o=read_default_label_raw(obj_dir_file)
                #data,_,_=read_txt(data_obj_file)
                loc_one=[]
                for i in range(len(o)):
                    y_polar=(o[i][0]+o[i][1])/2.0
                    x_polar=(o[i][2]+o[i][3])/2.0
                    y=np.sin(np.deg2rad(y_polar))*x_polar*2.0
                    x=np.cos(np.deg2rad(y_polar))*x_polar*2.0
                    loc_single=[h[i],s[i],y,x,file,o[i]]
                    loc_one.append(loc_single)
                    count+=1
                loc.append(loc_one)
            #for i in range(len(loc)):
            #    print(loc[i])
            #print(" ")
            loc_all.append(loc)
    return loc_all,count
    
def generate_BBox(human,states,label,obj):
    merge_label=[]
    for i in range(len(obj)):
        iou_max=0.0
        label_choose=-1
        for j in range(len(label)):
            iou_temp=calculate_iou(obj[i],label[j])
            if iou_max<iou_temp:
                iou_max=iou_temp
                label_choose=j
        if label_choose!=-1:
            #label_new_single=[human[label_choose],states[labeSl_choose],label[label_choose][0],label[label_choose][1],label[label_choose][2],label[label_choose][3]]
            #print(label_new_single)
            label_y=(label[label_choose][0]+label[label_choose][1])/2.0
            label_x=(label[label_choose][1]+label[label_choose][2])/2.0
            x=np.cos(np.deg2rad(label_y))*label_x*2.0
            y=np.sin(np.deg2rad(label_y))*label_x*2.0
            label_new_single=[human[label_choose],states[label_choose],label_y,label_x,y,x]
            merge_label.append(label_new_single)
    merge_label.sort(key=lambda x: np.int32(x[0]))
    localize=[]
    return merge_label

def motion_detection(label,data_obj):
    human_list=[]
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j][0] not in human_list:
                human_list.append(label[i][j][0])
    human_list.sort(key=lambda x:x)

    dict_loc={}
    for i in range(len(human_list)):
        dict_loc.update({human_list[i]:[]})
    time=2.5
    for i in range(len(label)):
        for j in range(len(label[i])):
            #print(label[i][j])
            dict_loc[label[i][j][0]].append([np.float32(label[i][j][2]),np.float32(label[i][j][3]),np.float32(time*i),label[i][j][4],label[i][j][5],label[i][j][0]])
    return dict_loc
    
def obtain_local_loc_center(label):
    x_loc=0.0
    y_loc=0.0
    number=0
    for i in range(len(label)):
        x_loc+=label[i][0]
        y_loc+=label[i][1]

def local_replish(label):
    length=len(label)
    #print(" ")
    #print(label)
    label.sort(key=lambda x:x[2])
    for i in range(1,length):
        if np.float32(label[i][2])-np.float32(label[i-1][2])>3:
            number=np.int32((label[i][2]-label[i-1][2])/2.5)
            #print(number)
            for j in range(1,number):
                x_add=label[i][1]*j/number+label[i-1][1]*(number-j)/number
                y_add=label[i][0]*j/number+label[i-1][0]*(number-j)/number
                data_single=[y_add,x_add,label[i-1][2]+2.5*j,label[i-1][3],label[i-1][4]]
                label.append(data_single)
    #print(" ")
    #print(label)
    return label

def moving_center_dis(move):
    x_c=0
    y_c=0
    for i in range(len(move)):
        x_c+=move[i][1]
        y_c+=move[i][0]
    x_c/=len(move)
    y_c/=len(move)
    #print(y_c,x_c)
    dis=np.sqrt((move[-1][0]-y_c)**2+(move[-1][1]-x_c)**2)
    dis_count=np.sqrt((move[-1][0]-move[0][0])**2+(move[-1][1]-move[0][1])**2)
    return dis,dis_count
    
def motion_loc_pred(loc_dict):
    fail_dirct=[]
    success_dir=[]
    time_threshold=7.5
    start_time=0
    count_correct=0
    state_dic={}
    for key in loc_dict:
        state_dic[key]=[]
    
        
    for key in loc_dict:
        moving_list=[]
        for i in range(len(loc_dict[key])):
            #print(loc_dict[key][i])
            if moving_list==[]:
                start_time=loc_dict[key][i][2]
                moving_list.append([loc_dict[key][i][0],loc_dict[key][i][1],loc_dict[key][i][2],loc_dict[key][i][3],loc_dict[key][i][4]])
            elif loc_dict[key][i][2]-start_time<=time_threshold:
                moving_list.append([loc_dict[key][i][0],loc_dict[key][i][1],loc_dict[key][i][2],loc_dict[key][i][3],loc_dict[key][i][4]])
            else:
                while loc_dict[key][i][2]-start_time>time_threshold:
                    #print(moving_list)
                    if moving_list!=[]:
                        moving_list.pop(0)
                    if moving_list!=[]:
                        start_time=moving_list[0][2]
                    else:
                        start_time=loc_dict[key][i][2]
                if moving_list==[]:
                    start_time=loc_dict[key][i][2]
                    moving_list.append([loc_dict[key][i][0],loc_dict[key][i][1],loc_dict[key][i][2],loc_dict[key][i][3],loc_dict[key][i][4]])
                else:
                    moving_list.append([loc_dict[key][i][0],loc_dict[key][i][1],loc_dict[key][i][2],loc_dict[key][i][3],loc_dict[key][i][4]])
            if len(moving_list)<=1:
                fail_dirct.append(loc_dict[key][i])
                state_dic[key].append(['non-moving',loc_dict[key][i][3]])
            else:
                #print(moving_list)
                #print(moving_list[-1][4],moving_list[-2][4])
                iou_s=calculate_iou_on_small(loc_dict[key][i][4],moving_list[-2][4])
                iou=calculate_iou(loc_dict[key][i][4],moving_list[-2][4])
                #print(moving_list[-1][4],moving_list[-2][4])
                #print(iou)
                #moving_list=local_replish(moving_list)
                dis,dis_count=moving_center_dis(moving_list)
                #print(dis,dis_count)
                if dis==0.0:
                    dis_ratio=0.0
                else:
                    dis_ratio=dis_count/dis
           
                if (iou>0.5 or iou_s>0.5) and (dis<30 or dis_count<30):
                    fail_dirct.append(loc_dict[key][i])
                    state_dic[key].append(['non-moving',loc_dict[key][i][3]])
                    #ßprint(dis,dis_count)
                else:   
                    if dis>60 or (dis_count>60 and dis_ratio>1.0): #or (dis>10 and dis_ratio>1.0): #or dis_ratio>2:
                        count_correct+=1
                        success_dir.append(loc_dict[key][i])
                        state_dic[key].append(['moving',loc_dict[key][i][3]])
                        #print(dis,dis_count)
                    else:
                        #print(dis,dis_count)
                        fail_dirct.append(loc_dict[key][i])
                        state_dic[key].append(['non-moving',loc_dict[key][i][3]])
        for key in state_dic:
            state_dic[key].sort(key=lambda x:np.float32(x[1][:-4].split("_")[1]))

    return count_correct,fail_dirct,success_dir,state_dic

    
def read_data(path_obj,path_label):
    h,s,obj=read_yolo_label(path_label)
    obj_de=read_BBox_data(path_obj)
    new_obj=generate_BBox(h,s,obj,obj_de)
    return new_obj

def save_results(save_path,obj_new):
    f=open(save_path,"w")
    for i in range(len(obj_new)):
        obj_single=str(obj_new[i][0])+" "+str(obj_new[i][1])+" "+str(obj_new[i][2])+" "+str(obj_new[i][3])+" "+str(obj_new[i][4])+" "+str(obj_new[i][5])+" \n"
        f.writelines(obj_single)
    f.close()

def process_data(obj_dir,label_dir):
    files=os.listdir(obj_dir)
    files_label=os.listdir(label_dir)
    files.sort(key=lambda x: x[:,-4].split("_")[1])
    files_label.sort(key=lambda x: x[:,-4].split("_")[1])
    obj_new=[]
    for i in range(len(files)):
        file_path=obj_dir+"/"+files[i]
        file_path_label=label_dir+"/"+files_label[i]
        new_obj_single=read_data(file_path,file_path_label)
        obj_new.append(new_obj_single)
    #print(new_obj_single)
    return obj_new
   
        