import sys
sys.path.append('..')
#from data.prepare_data import read_txt
from sklearn.cluster import DBSCAN
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
import cv2
import copy
#import denoise_metirc as DM


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

def read_txt(path, angle_range=400):
    ''' read sonar data from txt file
    Args:
        path: path of txt file
        angle_range: range of angle (1 gradian = 0.9 degree), default: 400
    Returns:
        sonar_data: 2d array, shape: (angle_range, 500)
        start_angle: start angle of sonar data
        end_angle: end angle of sonar data
    '''
    sonar_data = np.zeros((angle_range, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            if len(data) == 500:
                sonar_data[int(angle)] = data
    return sonar_data, int(start_angle), int(end_angle)

def read_img(path):
    data=cv2.imread(path)
    #(data.shape)
    return data

def read_default_label(file_path):
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
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[2]))
                ymin = int(float(arr[3]))
                xmax = int(float(arr[4]))
                ymax = int(float(arr[5]))
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[1]))
                ymin = int(float(arr[2]))
                xmax = int(float(arr[3]))
                ymax = int(float(arr[4]))

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def sonar2pos(sonar_data, threshold: int = 50):
    """
    Transform sonar data to point location, which can be used in cluster algorithm
    :param sonar_data: One sonar picture.
    :param threshold: The lowest strength of point.
    :return:
    """
    pos_arr = np.where(sonar_data >= threshold)
    num = pos_arr[0].shape[0]
    pos_matrix = []
    for i in range(num):
        pos_matrix.append([pos_arr[0][i], pos_arr[1][i]])
    pos_matrix = np.array(pos_matrix)
    return pos_matrix

def fuse_objects(denoised_objs, raw_objs, iou_threshold=0.8):
    ''' Denoised objects are more accurate, but smaller than raw objects.
        So use denoised objects to filter raw objects.
    '''
    if len(denoised_objs) == 0:
        return []
    if len(raw_objs) == 0:
        return denoised_objs
    fused_objs = []
    for denoised_obj in denoised_objs:
        for raw_obj in raw_objs:
            if calculate_iou_on_obj1(denoised_obj, raw_obj) > iou_threshold:
                fused_objs.append(raw_obj)
    if len(fused_objs) == 0:
        fused_objs = denoised_objs
    return fused_objs

def fuse_denoised_objects(denoised_objs, raw_objs, iou_threshold=0.4):
    if len(denoised_objs) == 0:
        return raw_objs
    fused_objs = []
    for denoised_obj in denoised_objs:
        for raw_obj in raw_objs:
            if calculate_iou_on_obj1(raw_obj,denoised_obj) > iou_threshold:
                fused_objs.append(raw_obj)
            elif calculate_iou_on_obj1(denoised_obj,raw_obj) > iou_threshold:
                fused_objs.append(denoised_obj)
    if len(fused_objs)==0:
        fused_objs = raw_objs
    return fused_objs
    
def fuse_BBox_objects(Obj_list,aver_list):
    if len(aver_list)==0 or (Obj_list)==0:
        return Obj_list
    final_obj=[]
    for obj_de in Obj_list:
        max_iou=0.0
        temp_obj=[]
        for obj in aver_list:
            temp_iou=calculate_iou_on_obj1(obj,obj_de)
            if temp_iou>max_iou and temp_iou>0.1:
                temp_obj=obj
                max_iou=temp_iou
        if temp_obj!=[]:
            final_obj.append(temp_obj)
            aver_list.remove(temp_obj)
        else:
            final_obj.append(obj_de)
    return final_obj
        

def calculate_iou_on_obj1(obj1, obj2):
    ''' Calculate the iou of two objects.
        obj1 should be the denoised object.
    '''
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    obj1_area = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2])
    iou = inter / obj1_area
    return iou

def proposed_denoise(data,save_dir,name_prefix,save_path):
    data_median=data.astype(np.uint8)
    data_aver=np.float32(data)
    data_denoise=cv2.medianBlur(data_median,17)
    data_localize=cv2.blur(data_aver,(5,5))
    objs_me=localize_one_pic_without_denoise(data_denoise)
    objs_aver=localize_one_pic_without_denoise(data_localize)
    objs_fused=fuse_objects(objs_aver,objs_me,0.5)
    save_file(save_dir,name_prefix,save_path,objs_fused)

def baseline_denoise(data,parameters,method='aver'):
    if method=='Median':
        data=data.astype(np.uint8)
    else:
        data=np.float32(data)
    if method=='aver':
        return cv2.blur(data,parameters[0])
    elif method=='guassian':
        return cv2.GaussianBlur(data,parameters[0],parameters[1])
    elif method=='bi':
        return cv2.bilateralFilter(data,parameters[0],parameters[1],parameters[2])
    elif method=='Median':
        return cv2.medianBlur(data,parameters[0])
    else:
        return data
    
def background_remove(bg,bg_re,sonar_data):
    clean=sonar_data-bg
    clean[clean<0]=0
    clean_re=sonar_data-bg_re
    clean_re[clean_re<0]=0
    if np.sum(clean_re)>np.sum(clean):
        return clean
    else:
        return clean_re
    
def only_localize(data,save_dir,name_prefix,save_path):
    objs=localize_one_pic_without_denoise(data)
    save_file(save_dir,name_prefix,save_path,objs)
    
def denoise_localization_save(data,save_dir,name_prefix,save_path,parameters,method):
    data_processed=baseline_denoise(data,parameters,method)
    objs=localize_one_pic_without_denoise(data_processed)
    save_file(save_dir,name_prefix,save_path,objs)
    
def localization_save(data,save_dir,name_prefix,save_path):
    save_path=save_path[:-4]+".txt"
    objs=localize_one_pic_without_denoise(data)
    save_file(save_dir,name_prefix,save_path,objs)
            
def save_file(dir,save_name_prefix,save_path,objects_data):
    f=open(dir+"/"+save_name_prefix+"_"+save_path,'w')
    for j in range(len(objects_data)):
        obj=objects_data[j]
        obj_re=str(obj[2])+","+str(obj[0])+","+str(obj[3])+","+str(obj[1])+","+"\n"
        f.writelines(obj_re)
    f.close()
    
def localize_one_pic_aver(temp_data, threshold=45, min_samples=10, eps=12, blur_size=5, min_size=25):
    """
    Transform sonar data to point location, which can be used in cluster algorithm
    :param temp_data: One sonar picture.
    :param threshold: The lowest strength of point for filter.
    :return:
    """
    temp_data = temp_data.astype(np.uint8)
    if blur_size > 0:
        temp_data = cv2.blur(temp_data, (blur_size,blur_size))
    
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    object_poses = []

    point_pos = sonar2pos(temp_data, threshold)
    try:
        labels = dbscan.fit_predict(point_pos)
    except:
        return object_poses
    label_class = np.unique(labels)
    for label in label_class:
        if label < 0:
            continue
        obj_pos = np.where(labels == label)[0]
        temp_dis = []
        temp_angle = []

        for point in obj_pos:
            temp_dis.append(point_pos[point, 1])
            temp_angle.append(point_pos[point, 0])
        min_dis = min(temp_dis)
        max_dis = max(temp_dis)
        min_angle = min(temp_angle)
        max_angle = max(temp_angle)
        if (max_angle - min_angle) * (max_dis - min_dis) < min_size:
            continue
        object_poses.append([min_angle, max_angle, min_dis, max_dis])
    return object_poses

def localize_one_pic_without_denoise(temp_data,threshold=45, min_samples=10, eps=12, min_size=25):
    temp_data = temp_data.astype(np.uint8)
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    object_poses = []

    point_pos = sonar2pos(temp_data, threshold)
    try:
        labels = dbscan.fit_predict(point_pos)
    except:
        return object_poses
    label_class = np.unique(labels)
    for label in label_class:
        if label < 0:
            continue
        obj_pos = np.where(labels == label)[0]
        temp_dis = []
        temp_angle = []

        for point in obj_pos:
            temp_dis.append(point_pos[point, 1])
            temp_angle.append(point_pos[point, 0])
        min_dis = min(temp_dis)
        max_dis = max(temp_dis)
        min_angle = min(temp_angle)
        max_angle = max(temp_angle)
        if (max_angle - min_angle) * (max_dis - min_dis) < min_size:
            continue
        object_poses.append([min_angle, max_angle, min_dis, max_dis])
    return object_poses
    
    
def localize_one_pic(temp_data, threshold=45, min_samples=10, eps=12, blur_size=5, min_size=25):
    """
    Transform sonar data to point location, which can be used in cluster algorithm
    :param temp_data: One sonar picture.
    :param threshold: The lowest strength of point for filter.
    :return:
    """
    temp_data = temp_data.astype(np.uint8)
    if blur_size > 0:
        temp_data = cv2.medianBlur(temp_data, blur_size)
    
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    object_poses = []

    point_pos = sonar2pos(temp_data, threshold)
    try:
        labels = dbscan.fit_predict(point_pos)
    except:
        return object_poses
    label_class = np.unique(labels)
    for label in label_class:
        if label < 0:
            continue
        obj_pos = np.where(labels == label)[0]
        temp_dis = []
        temp_angle = []

        for point in obj_pos:
            temp_dis.append(point_pos[point, 1])
            temp_angle.append(point_pos[point, 0])
        min_dis = min(temp_dis)
        max_dis = max(temp_dis)
        min_angle = min(temp_angle)
        max_angle = max(temp_angle)
        if (max_angle - min_angle) * (max_dis - min_dis) < min_size:
            continue
        object_poses.append([min_angle, max_angle, min_dis, max_dis])
    return object_poses
    
def process_one_pic(sonar_data, occupancy=False):
    ''' 
    Use different parameters to localize objects in one pic
    :param sonar_data: One sonar data.
    :return: object_poses: A list of objects' poses. Each object is a list of [min_angle, max_angle, min_dis, max_dis]
    '''
    # Get denoised bbox
    denoise_configs = {
        'denoise1' : {
            # general
            'threshold': 40,
            'min_samples': 20,
            'eps': 15,#15
            'blur_size': 5,
            'min_size': 15,
        },
        'denoise2' : {
            # for weak objects
            'threshold': 30,
            'min_samples': 20,
            'eps': 17,#17
            'blur_size': 5,#9
            'min_size': 15,#20
        },
        'denoise3' : {
            # for strong noise
            'threshold': 40,
            'min_samples': 20,
            'eps': 10,#10
            'blur_size': 17,#13
            'min_size': 15,
        },
    }
    raw_configs = {
        'raw1' : {
            # general
            'threshold': 60,
            'min_samples': 15,
            'eps': 17,
            'blur_size': 0,
            'min_size': 30,
        },
        'raw2' : {
            # for weak objects
            'threshold': 30,
            'min_samples': 15,
            'eps': 15,
            'blur_size': 0,
            'min_size': 35,
        },
        'raw3' : {
            # for weak objects
            'threshold': 20,
            'min_samples': 15,
            'eps': 20,
            'blur_size': 0,
            'min_size': 45,
        },
    }
    
    aver_configs={
        'aver' : {
            # for weak objects
            'threshold': 30,
            'min_samples': 15,
            'eps': 17,
            'blur_size': 13,
            'min_size': 45,
        },
    }
    
    denoised_object_poses = []
    raw_object_poses = []
    sonar_data = copy.deepcopy(sonar_data)
    #sonar_data[:, :60] = 0
    denoised_object_poses = localize_one_pic(sonar_data, **denoise_configs['denoise1'])
    if len(denoised_object_poses) == 0:
        denoised_object_poses = localize_one_pic(sonar_data, **denoise_configs['denoise2'])

    for raw_config in raw_configs.values():
        raw_object_poses = localize_one_pic(sonar_data, **raw_config)
        if len(raw_object_poses) > 0:
            break

    object_poses = fuse_objects(denoised_object_poses, raw_object_poses)

    result_poses = []
    K_size=5
    while obj_physical_filter(object_poses,18000,120,300): #rescale,size enlarge.
        K_size+=2
        denoise_configs['denoise2']['blur_size']=K_size
        denoised_object_poses = localize_one_pic(sonar_data, **denoise_configs['denoise2'])
        #for raw_config in raw_configs.values():
        #    raw_object_poses = localize_one_pic(sonar_data, **raw_config)
        #    if len(raw_object_poses) > 0:
        #        break
        #object_poses = fuse_objects(denoised_object_poses, raw_object_poses)
        if K_size==17:
            break
        
    objs_well_denoise= localize_one_pic(sonar_data,**denoise_configs['denoise3'])
    
    object_poses=fuse_denoised_objects(objs_well_denoise,object_poses)
    obj_idx=0
    while obj_idx<len(object_poses):
        if obj_tiny_filter(object_poses[obj_idx]):
            object_poses.remove(object_poses[obj_idx])
            obj_idx-=1
        obj_idx+=1
    #aver_obj=localize_one_pic_aver(sonar_data,**aver_configs['aver'])
    
    #object_poses=fuse_BBox_objects(object_poses,aver_obj)
    
    for obj in object_poses:
        result_poses.append(obj)
        #pass
    '''
    for obj in object_poses:
        # if any obj size is too large, use denoise3 in this huge object
        if (obj[1] - obj[0]) * (obj[3] - obj[2]) > 2500 or (obj[3] - obj[2]) > 100:
            pivot_x = obj[2]
            pivot_y = obj[0]
            obj_data = sonar_data[int(obj[0]):int(obj[1]), int(obj[2]):int(obj[3])]
            temp_poses = localize_one_pic(obj_data, **denoise_configs['denoise3'])

            for i, obj in enumerate(temp_poses):
                result_poses.append([max(0, obj[0] + pivot_y - 5), 
                                        min(obj[1] + 5 + pivot_y, 400), 
                                        max(obj[2] - 5 + pivot_x, 0), 
                                        min(obj[3] + 5 + pivot_x, 500)])
        else:
            result_poses.append(obj)
    '''
    if len(result_poses) == 0:
        result_poses = denoised_object_poses
    if len(result_poses) == 0:
        result_poses = raw_object_poses
    return result_poses

def obj_physical_filter(obj_list, human_size, angle_thre, length_threshold):
    for obj in obj_list:
        if(obj[1]-obj[0])*(obj[3]-obj[2])>human_size or np.abs(obj[1]-obj[2])>angle_thre or np.abs(obj[3]-obj[2])>length_threshold:
            return True
    return False

def obj_tiny_filter(obj,human_size=120,angle_thre=5,length_threshold=10):
    if(obj[1]-obj[0])*(obj[3]-obj[2])<human_size or np.abs(obj[1]-obj[2])<angle_thre or np.abs(obj[3]-obj[2])<length_threshold:
        return True
    return False

def process_one_scene(data_path, pic_save_root):
    id2state = {
            1: 'stand',
            2: 'struggle',
            3: 'sink',
            4: 'float',
        }
    basename = os.path.basename(os.path.split(data_path)[-2])
    state = id2state.get(int(basename[-3]), 'unknown')
    files = os.listdir(data_path)
    for i, filename in enumerate(files):
        sonar_data, start_angle, end_angle = read_txt(os.path.join(data_path, filename))
        # process one pic
        object_poses = process_one_pic(sonar_data)
        # temp_data = copy.deepcopy(sonar_data)
        # temp_data[:, :60] = 0
        # object_poses = localize_one_pic(temp_data)
        # save pic and label
        im = Image.fromarray(sonar_data)
        im = im.convert('RGB')

        pic_save_path = os.path.join(pic_save_root, filename[:-4] + '-raw.jpg')
        os.makedirs(os.path.dirname(pic_save_path), exist_ok=True)
        im.save(pic_save_path)

        for i, re in enumerate(object_poses):
            a = ImageDraw.ImageDraw(im)
            a.rectangle(((re[2], re[0]), (re[3], re[1])), fill=None, outline='red', width=1)
        pic_save_path = os.path.join(pic_save_root, filename[:-4] + '.jpg')
        os.makedirs(os.path.dirname(pic_save_path), exist_ok=True)
        im.save(pic_save_path)

        # label_save_path = os.path.join(label_save_root, filename[:-4] + '.txt')
        # os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
    
        # with open(label_save_path, 'w') as f:
        #     for i, obj in enumerate(object_poses):
        #         f.write(str(0) + ' ' + str(state) + ' ' + str(obj[2]) + ' ' + str(obj[0]) + ' ' + str(obj[3]) + ' ' + str(obj[1]) + '\n')



