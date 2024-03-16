import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
matplotlib.rc('pdf', fonttype=42)
plt.rc('font', family='serif', size=12)
def read_state_list(path):
    f=open(path,'r')
    lines=f.readlines()
    #print(type(lines))
    state=[]
    real=[]
    label=[]
    for line in lines:
        data=line.split(",")
        state.append([np.float32(data[0]),np.float32(data[1])])
        real.append(np.float32(data[2]))
        label.append(np.float32(data[3][0]))
    state=np.array(state)
    real=np.array(real)
    label=np.array(label)
    return state,real,label

def SSE(location):
    mean_pos=[0,0]
    for i in range(len(location)):
        mean_pos[0]+=location[i][0]
        mean_pos[1]+=location[i][1]
    mean_pos[0]/=len(location)
    mean_pos[1]/=len(location)
    #SSE_loss=0.0
    SSE_array=np.zeros(len(location))
    count=0
    for i in range(len(location)):
        #SSE_loss+=np.abs(np.sqrt((location[i][0]-mean_pos[0])**2+(location[i][1]-mean_pos[1])**2))
        SSE_array[i]=np.abs(np.sqrt((location[i][0]-mean_pos[0])**2+(location[i][1]-mean_pos[1])**2))
        count+=1
    SSE_loss_var=np.std(SSE_array)
    SSE_loss_mean=np.mean(SSE_array)
    return SSE_loss_mean,SSE_loss_var
    

class swimmer_state:
    def __init__(self,state_list,length):
        self.motion=[]                                                          
        self.location=[]
        #self.last_motion=self.motion[-1]
        self.motion_score=[]
        self.frequency_score=[]
        self.state_list=state_list
        self.timestamp=[]
        self.length=length
        self.duration=[]
        self.list_full=False
        self.state_transfer_machine=None
        #self.state_list=state_list
        self.location_info=[]
        self.SSE_mean=[]
        self.SSE_var=[]
            
    def update_move(self,motion,location,timestamp):
        self.motion.append(motion)
        self.location.append(location)
        self.timestamp.append(timestamp)
        
    def state_smooth(self,windows,step):
        motion_filter=[]
        re_motion=np.array(self.motion)
        for i in range(0,len(self.motion),step):
            if i-np.int32(windows/2.0)<0 or i-np.int32(windows/2.0):
                motion_filter.append(self.motion[i])
            else:
                data=[0,0]
                data[0]=np.mean(re_motion[i-np.int32(windows/2.0):i-np.int32(windows/2.0)][0])
                data[1]=np.mean(re_motion[i-np.int32(windows/2.0):i-np.int32(windows/2.0)][1])
                motion_filter.append(data)
                
        self.motion=motion_filter
        
    def state_smooth_vote(self):
        pass
    
    def update_timestamp(self,time):
        self.timestamp.append(time)
        while len(self.timestamp)>self.length:
            self.timestamp.remove()
        
    def update_state(self,state):
        self.state_list.append(state)
        while len(self.state_list)>self.length:
            self.state_list.remove(self.state_list[0])
    
    def update_motion_score(self):
        still_conf=self.motion[-1][0]
        motion_conf=self.motion[-1][1]
        diff=still_conf-motion_conf
        still_score=(1.0+diff)/2
        motion_score=1-still_score
        self.motion_score=[still_score,motion_score]
        return self.motion_score
        
    def update_location_score(self,times=5.0):
        if len(self.location)>1:
            diff_loc=self.location[len(self.location)-1]-self.location[len(self.location)-2]
            time_late=self.timestamp[len(self.location)-1]
            time_early=self.timestamp[len(self.location)-2]
            velocity=diff_loc[len(diff_loc)-1]/(time_late-time_early)
            distance=np.sqrt((self.location[len(self.location)-1][0]-self.location[len(self.location)-2][0])**2+(self.location[len(self.location)-1][1]-self.location[len(self.location)-2][1])**2)
            location_dict={}
            location_dict.update({"loc_diff":diff_loc})
            location_dict.update({"velocity":velocity})
            location_dict.update({"distance":distance})
            self.location_info.append(location_dict)
            #time_cur=self.timestamp[0]
            index_time=len(self.timestamp)-1
            while index_time>0:
                if self.timestamp[len(self.timestamp)-1]-self.timestamp[index_time]<times:
                    index_time-=1
                else:
                    break
            SSE_loss_mean,SSE_loss_var=SSE(self.location[index_time:len(self.timestamp)])
            #cur_index=0
            self.SSE_mean.append(SSE_loss_mean)
            self.SSE_var.append(SSE_loss_var)
        else:
            location_dict={}
            location_dict.update({"loc_diff":[0,0]})
            location_dict.update({"velocity":0})
            location_dict.update({"distance":0})
            self.location_info.append(location_dict)
            self.SSE_mean.append(0)
            self.SSE_var.append(0)
        while self.length<len(self.location):
            self.location_info.remove(self.location_info[0])
            self.location.remove(self.location[0])
            self.timestamp.remove(self.timestamp[0])
            self.SSE_mean.remove(self.SSE_mean[0])
            self.SSE_var.remove(self.SSE_var[0])
            
            
    def update_motion_frequency(self):
        Motion_list=[]
        F=0
        time_start=self.timestamp[0]
        time_end=0.0
        time_list=[]
        #print(self.timestamp)
        if len(self.motion)<=self.length:
            for i in range(len(self.motion)):
                if i <= len(self.motion)-2:
                    if self.motion[i][1]==1 and self.motion[i+1][1] == 0:
                        Motion_list.append(F+1)
                        time_end=self.timestamp[i]
                        time_list.append(time_end-time_start)
                        #print(time_end,time_start)
                        F=0
                    elif self.motion[i][1]==1 and self.motion[i+1][1]==1:
                        F+=1
                    elif self.motion[i][1]==0 and self.motion[i+1][1]==1:
                        F=1
                        time_start=self.timestamp[i]
                    else:
                        continue
        else:
            for i in range(len(self.motion)-self.length,self.length):
                if i <= len(self.motion)-2:
                    if self.motion[i][1]==1 and self.motion[i+1][1] == 0:
                        Motion_list.append(F+1)
                        time_end=self.timestamp[i]
                        time_list.append(time_end-time_start)
                        F=0
                    elif self.motion[i][1]==1 and self.motion[i+1][1]==1:
                        F+=1
                    elif self.motion[i][1]==0 and self.motion[i+1][1]==1:
                        F=1
                        time_start=self.timestamp[i]
                    else:
                        continue
        #print(self.motion)
        #print(Motion_list)
        if self.motion[-1][1]==1:
            Motion_list.append(F)
            time_list.append(self.timestamp[-1]-time_start)
        #print(time_list)
        if len(Motion_list)>0:
            frequency_dict={}
            frequency_dict.update({"frequency_score_list":Motion_list})
            frequency_dict.update({"latest":Motion_list[-1]})
            Motion_list=np.array(Motion_list)
            F_mean=np.mean(Motion_list)
            F_var=np.var(Motion_list)
            #count=Motion_list.size
            frequency_dict.update({"average":F_mean})
            frequency_dict.update({"var":F_var})
            frequency_dict.update({"len":len(self.motion)})
            frequency_dict.update({"latest_time":time_list[-1]})
            self.frequency_score.append(frequency_dict)
        else:
            frequency_dict={}
            frequency_dict.update({"frequency_score_list":Motion_list})
            frequency_dict.update({"latest":0})
            Motion_list=np.array(Motion_list)
            F_mean=np.mean(Motion_list)
            F_var=np.var(Motion_list)
            #count=Motion_list.size
            frequency_dict.update({"average":F_mean})
            frequency_dict.update({"var":F_var})
            frequency_dict.update({"len":len(self.motion)})
            frequency_dict.update({"latest_time":0})
            self.frequency_score.append(frequency_dict)
        while len(self.frequency_score)>self.length:
            self.frequency_score.remove(self.frequency_score[0])
        
    def update_duration(self):
        #print(self.state_list)
        if len(self.state_list)>=2:
            count=1
            length=len(self.state_list)
            for i in range(1,length):
                if i!=length and self.state_list[length-i]==self.state_list[length-i-1]:
                    count+=1
                else:
                    #count=0
                    break
            #print(count,self.timestamp)
            self.duration.append([self.timestamp[len(self.timestamp)-1]-self.timestamp[len(self.timestamp)-count],length])
            #print(self.duration)
        elif len(self.state_list)==1:
            self.duration.append([1,len(self.state_list)])
        else:
            self.duration.append([0,len(self.state_list)])
        #print(self.duration)
        while len(self.duration)>self.length:
            self.duration.remove(self.duration[0])
        
    def update(self,motion,location,timestamp):
        self.update_move(motion,location,timestamp)
        self.update_motion_score()
        self.update_location_score()
        self.update_motion_frequency()
        self.update_duration()
        if self.state_transfer_machine==None:
            self.state_transfer_machine=state_transfer()
        state=self.state_transfer_machine.transfer_state()
        self.state_list.append(state)
        while len(self.state_list)>self.length:
            self.state_list.remove(self.state_list[0])
        
    def simulation_update(self,state_new,timestamp):
        new_state=[0,0]
        if state_new[0]=='S':
            new_state=[1,0]
        else:
            new_state=[0,1]
        self.motion.append(new_state)
        self.timestamp.append(timestamp)
        while len(self.motion)>self.length:
            self.motion.remove(self.motion[0])
            self.timestamp.remove(self.timestamp[0])
        self.update_motion_frequency()
        self.update_duration()
        return self.frequency_score,self.duration
    

class state_transfer:
    def __init__(self,cur_state, motion_score=None, location_info=None, duration=None, frequency=None, state=['moving','still','patting','struggling','drowning']):
        self.swim_state=state
        self.motion_score=motion_score
        self.localtion_score=location_info
        self.frequency=frequency
        self.duration=duration
        self.cur_state=cur_state
        self.dronwing_mark={
            "Motion":['S','M'],
            "Location":['U','C'],
            "Frequency":['S','F'],
            #"Velocity":['F','S'],
            "Duration":['S','L']
        }
        self.weight=[0.25,0.25,0.25,0.25]
        self.state_list=state
    
    def update_cur_state(self,state):
        self.cur_state=state
        
    def transfer_state_mark(self,simulator=False, action=None):
        if simulator:
            state_new=action
        else:
            state_new=[self.check_motion_state,self.check_location_changed,self.check_frequency_switch,self.check_state_duration]
        #print(action,self.cur_state)
        #state_re=None
        #print(self.cur_state,self.state_list[0])
        #print(state_new)
        if self.cur_state==None:
            if state_new[0]=='M':
                return self.state_list[2]
            else:
                return self.state_list[1]
        
        if self.cur_state==self.state_list[0] or self.cur_state==self.state_list[1]:
            if state_new[1]=='C':
                return self.state_list[0]
            elif state_new[0]=='M' and state_new[1]=='U':
                return self.state_list[2]
            elif state_new[0]=='S' and state_new[1]=='U':
                if state_new[0]=='S' and state_new[3]=='L':
                    return self.state_list[4]
                return self.state_list[1]
            else:
                return self.cur_state
        
        if self.cur_state==self.state_list[2]:
            if state_new[1]=='C':
                return self.state_list[0]
            elif state_new[0]=='M' and state_new[1]=='U' and state_new[2]=="S":
                return self.state_list[3]
            elif state_new[0]=='M' and state_new[1]=='U':
                return self.state_list[2]
            elif state_new[0]=='S' and state_new[1]=='U':
                return self.state_list[1]
            else:
                return self.cur_state
            
        if self.cur_state==self.state_list[3]:
            if state_new[3]=="L":
                return self.state_list[4]
            else:
                return self.cur_state
            
        if self.cur_state==self.state_list[4]:
            return self.cur_state
        
        
    def check_motion_state(self):
        motion_score=self.motion_score[-1]
        if motion_score[1]==1:
            return self.dronwing_mark['motion'][1]
        else:
            return self.dronwing_mark['motion'][0]
        
    def check_frequency_switch(self):

        frequency_score=self.frequency[-1]
        #print(frequency_score)
        '''
        frequency_score_ratio=frequency_score['latest']/frequency_score['len']        
        #print(frequency_score_ratio)
        if frequency_score_ratio>0.7:
            #print(self.dronwing_mark["Frequency"][0])
            return self.dronwing_mark["Frequency"][0]
        else:
            #print(self.dronwing_mark["Frequency"][1])
            return self.dronwing_mark["Frequency"][1]
        '''
        frequency_score_ratio=frequency_score['latest']#/frequency_score['len']        
        #print(frequency_score_ratio)
        if frequency_score_ratio>=8: #for current sensing parameters, can be modified, in experiments, the length of motion is fixed.
            #print(self.dronwing_mark["Frequency"][0])
            return self.dronwing_mark["Frequency"][0]
        else:
            #print(self.dronwing_mark["Frequency"][1])
            return self.dronwing_mark["Frequency"][1]
        
    def check_state_duration(self):
        duration_info=self.duration[-1]
        '''
        if duration_info[1]!=0:
            duration_potential=duration_info[0]/duration_info[1]
        else:
            duration_potential=0.0
        #print(duration_potential)
        if duration_potential>0.8:
            return self.dronwing_mark["Duration"][1]
        else:
            return self.dronwing_mark["Duration"][0]
        '''
        #print(duration_potential)
        if self.cur_state=="still":
            if duration_info[0]>=60: #only for experiemnts results, can set manually
                return self.dronwing_mark["Duration"][1]
            else:
                return self.dronwing_mark["Duration"][0]
        else:
            if duration_info[0]>20:
                return self.dronwing_mark["Duration"][1]
            else:
                return self.dronwing_mark["Duration"][0]
        
    def check_location_changed(self):
        location_ve=self.localtion_score['velocity']
        location_info=self.localtion_score['distance']
        if location_info>2.5 or location_ve>1.0:
            return self.dronwing_mark['Location'][1]
        else:
            return self.dronwing_mark['Location'][0]
        
def sample(time_stamp,state,scan_time,GT):
    new_state=[]
    new_time=[]
    new_GT=[]
    count=1
    new_state.append(state[0])
    new_time.append(time_stamp[0])
    new_GT.append(GT[0])
    for i in range(1,len(time_stamp)):
        if time_stamp[i-1]<=count*scan_time and time_stamp[i]>=count*scan_time:
            new_time.append(time_stamp[i])
            new_state.append(state[i])
            new_GT.append(GT[i])
            count+=1
    return new_state,new_time,new_GT

def metrics(GT,detect):
    count=0
    correct=0
    for i in range(len(GT)):
        if GT[i]==detect[i]:
            correct+=1
        count+=1
    ratio=correct*1.0/count
    return correct,count,ratio

def metircs_class(GT,detect):
    state=['moving','standing','patting','struggling','drowning']
    dict_re=[[0,0],[0,0],[0,0],[0,0],[0,0]]
    count=0
    correct=0
    for i in range(len(GT)):
        if GT[i]==detect[i]:
            correct+=1
            dict_re[state.index(GT[i])][0]+=1
        count+=1
        dict_re[state.index(GT[i])][1]+=1
    ratio=correct*1.0/count
    return correct,count,ratio,dict_re
    
def confusion_metrix(GT,detect):
    state=['moving','standing','patting','struggling','drowning']
    detect_re=np.zeros((len(state),len(state)))
    #dict_re=[[0,0],[0,0],[0,0],[0,0],[0,0]]
    count=0
    correct=0
    for i in range(len(GT)):
        if GT[i]==detect[i]:
            correct+=1
            #dict_re[state.index(GT[i])][0]+=1
        detect_re[state.index(GT[i])][state.index(detect[i])]+=1
        count+=1
    ratio=correct*1.0/count
    #class_num=np.zeros(len(state))
    detect_re_per=np.zeros((len(state),len(state)))
    for i in range(len(detect_re)):
        class_num=np.sum(detect_re[i])
        for j in range(len(detect_re[i])):
            detect_re_per[i][j]=detect_re[i][j]/class_num
    return correct,count,ratio,detect_re,detect_re_per

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
    for i in range(np.shape(detect_re)[0]):
        for j in range(np.shape(detect_re)[1]):
            if detect_re[i][j]>0.5:
                plt.text(j,i,str(format(detect_re[i][j],'.3f')),ha="center",va="center",color="white")
            else:
                plt.text(j,i,str(format(detect_re[i][j],'.3f')),ha="center",va="center",color="black")
    plt.title(name)
    plt.show()
    #plt.savefig("../paper_figure/"+save_name+".pdf",bbox_inches = 'tight', dpi = 100)
    plt.close()
    

