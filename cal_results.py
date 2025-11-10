import numpy as np
import matplotlib.pyplot as plt

def positive_negative_cls(matrix):
    p_n_m=np.zeros((2,2))
    count_p_p=0
    for i in range(0,3):
        for j in range(0,3):
            count_p_p+=matrix[i][j]
    
    count_n_p=0
    for i in range(3,5):
        for j in range(0,3):
            count_n_p+=matrix[i][j]
    count_p_n=0
    for i in range(0,3):
        for j in range(3,5):
            count_p_n+=matrix[i][j]
    count_n_n=0
    for i in range(3,5):
        for j in range(3,5):
            count_n_n+=matrix[i][j]
    count_p=count_p_p+count_p_n
    count_n=count_n_n+count_n_p
    p_n_m[0][0]=count_p_p
    p_n_m[0][1]=count_p_n
    p_n_m[1][0]=count_n_p
    p_n_m[1][1]=count_n_n
    m_r=np.zeros((2,2))
    m_r[0][0]=count_p_p/count_p
    m_r[0][1]=count_p_n/count_p
    m_r[1][0]=count_n_p/count_n
    m_r[1][1]=count_n_n/count_n
    return p_n_m,m_r

def matrix2ratio(matrix):
    matrix_new=np.zeros((5,5))
    for i in range(len(matrix)):
        count=np.sum(matrix[i])
        for j in range(len(matrix[i])):
            if count!=0:
                matrix_new[i][j]=matrix[i][j]/count
            else:
                matrix_new[i][j]=0
    return matrix_new

def matrix2pn(matrix):
    matrix_new=np.zeros((2,2))
    for i in range(len(matrix)):
        count=np.sum(matrix[i])
        for j in range(len(matrix[i])):
            if count!=0:
                #print(matrix,count)
                matrix_new[i][j]=matrix[i][j]/count
            else:
                matrix_new[i][j]=0
    return matrix_new

def overall(martix):
    count_all=0
    correct=0
    for i in range(len(martix)):
        for j in range(len(martix[i])):
            count_all+=martix[i][j]
            if i==j:
                correct+=martix[i][j]
    return correct*1.0/count_all

def read_file_data(dirs,targets):
    data_detect_all=np.array([0,0,0,0])
    data_IoU_all=np.array([])
    dir_one=dirs[0]
    for target in targets:
        data_detect="./"+dir_one+"/"+target+"_detect.npy"
        data_iou="./"+dir_one+"/"+target+"_IoU.npy"
        data_detect_static=np.load(data_detect)
        data_IoU_static=np.load(data_iou)
        if data_IoU_all.size==0:
            data_IoU_all=data_IoU_static
        else:
            data_IoU_all=np.concatenate((data_IoU_all,data_IoU_static))
        for i in range(len(data_detect_static)):
            data_detect_all[i]+=data_detect_static[i]
    return data_detect_all,data_IoU_all

def cal_metric_direct(data_miss,data_wrong,data_all):
    wrong_all_num=0.0
    miss_all_num=0.0
    all_sub_num=0.0
    for i in range(len(data_wrong)):
        wrong_all_num+=data_wrong[i]
        miss_all_num+=data_miss[i]
        all_sub_num+=data_all[i]
    detect_all_num=all_sub_num-miss_all_num
    precision=detect_all_num/(detect_all_num+wrong_all_num)
    recall=detect_all_num/(detect_all_num+miss_all_num)
    F_score=2*precision*recall/(precision+recall)

def cal_metric(data_detect,data_IoU):
    miss_rate=data_detect[1]*1.0/data_detect[2]
    detected_obj=data_detect[2]-data_detect[1]
    precision=detected_obj/(detected_obj+data_detect[0])
    recall=detected_obj/(detected_obj+data_detect[1])
    F_score=2*precision*recall/(precision+recall)
    IoU_aver=np.mean(data_IoU)
    print("detect_results:")
    print("F1_score",F_score)
    print("miss_rate",miss_rate)
    print("IoU_aver",IoU_aver)

def draw_confusion_matirx(detect_re,classes,name,save_name="e2e"):
    fig,ax=plt.subplots()
    fig.set_size_inches(4, 3)
    plt.imshow(detect_re,cmap=plt.cm.Blues)
    indices=range(len(detect_re[0]))
    plt.xticks(indices,classes)
    plt.yticks(indices,classes)
    plt.colorbar()
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.xticks(rotation=30)
   # plt.yticks(rotation=330)
    two_detect=np.round(detect_re,2)
    print(two_detect)
    for i in range(np.shape(detect_re)[0]):
        for j in range(np.shape(detect_re)[1]):
            if detect_re[i][j]>0.5:
                plt.text(j,i,str(format(detect_re[i][j],'.2f')),ha="center",va="center",color="white")
            else:
                plt.text(j,i,str(format(detect_re[i][j],'.2f')),ha="center",va="center",color="black")
    plt.title(name)
    #plt.tight_layout()
    #plt.show()
    plt.savefig("./save_figure/"+save_name+".pdf",bbox_inches = 'tight',pad_inches=0)
    plt.savefig("./save_figure/"+save_name+".png",bbox_inches = 'tight',pad_inches=0)
    plt.close()
    print(name)

def main_object():
    dirs=["./object_detection_results"]
    target=["e2e_111"] #replace with save_results
    data_detect_all,data_IoU_all=read_file_data(dirs,target)
    cal_metric(data_detect_all,data_IoU_all)
    
def main_e2e_acc():
    dirs=["./acc_results"]
    target=["save_111"] #replace with save_results
    e2e_results=np.zeros((5,5))
    for i in range(len(target)):
        data=np.load(dirs[0]+"/"+target[i]+".npy")
        e2e_results+=data
    state=['Move','Motionless','Splash','Struggle','Drown']
    e2e_r=matrix2ratio(e2e_results)
    #print(e2e_results)
    print(overall(e2e_results))
    draw_confusion_matirx(e2e_r,state,"","e2e")
    e2e_p_n,e2e_p_n_r=positive_negative_cls(e2e_results)
    print(e2e_p_n)
    state_2cls=["Safe","Dangerous"]
    draw_confusion_matirx(e2e_p_n_r,state_2cls,"","e2e_2cls")
    
if __name__=="__main__":
    main_object()
    main_e2e_acc()
