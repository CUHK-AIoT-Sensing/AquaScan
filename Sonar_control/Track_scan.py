import numpy as np
import os
import random

def track(obj, id, time):
    pass
class swimmer():
    def __init__(self, positions,seg,state,state_score):
        self.last_obj=None
        self.speed=None
        self.positions=positions
        self.seg=seg
        self.state=state
        self.state_score=state_score
        self.direction=None
        self.swim_score=-1
        self.delta_direction=0
        self.delta_speed=0
        self.timestamp=0
        
    def update_last(self,last_obj):
        self.last_obj=last_obj
        
    def update_timestamp(self,time):
        self.timestamp=time
        
    def update_speed(self,time):
        self.speed=(self.positions-self.last_obj.positions)/time
        
    def update_direction(self):
        self.direction=np.arctan2(self.speed[1],self.speed[0])*180/np.pi
        
    def update_state(self,new_state):
        self.state=new_state
    
    def update_delta(self):
        if self.last_obj!=None:
            self.delta_direction=self.direction-self.last_obj.direction
            self.delta_speed=self.speed-self.last_obj.speed
        
    def obtain_state_score(self):
        if self.state=='struggle':
            self.swim_score=0.75+(1-self.state_score)/(100-1)*0.25
        elif self.state == 'float':
            self.swim_score=0.5+(1-self.state_score)/(100-1)*0.25
        elif self.state == 'sink':
            self.swim_score=(1-self.state_score)/(100-1)*0.25
        else:
            self.swim_score=0.25+(1-self.state_score)/(100-1)*0.25
        
    def obtain_data(self):
        return self.positions,self.seg
        
    def obtain_obj(self):
        return [self.positions,self.speed,self.swim_score,self.state,self.delta_direction,self.delta_speed]
        
    def update_all(self,last_obj,time,new_state,timestamp):
        self.update_obj(last_obj)
        self.update_speed(time)
        self.update_direction()
        self.update_state(state)
        self.update_delta()
        self.update_timestamp(timestamp)
        
class Swimmer_score():
    def __init__(self,cur_obj,last_obj,weight,virtue=False):
        self.score=0
        self.last_obj=last_obj
        self.cur_obj=cur_obj
        self.weight=weight
        self.motion_score=0
        self.state_score=0
        self.virtue=virtue
    
    def update_obj(self,new_obj):
        self.last_obj=self.cur_state
        self.cur_state=new_obj
    
    def update_score(self,new_obj):
        #if new_seg==None:
        update_obj(self,new_obj)
        #update_seg(self,new_seg)
        return overall_score(self)
        
    def motion_still(self,state):
        if state=="sink" or state=="stand":
            return 0
        else:
            return 1
            
    def state_compare(self):
        state_last=self.motion_still(self.last_obj[3])
        state_cur=self.motion_still(self.cur_obj[3])
        if state_last==state_cur:
            return 0
        elif state_last>state_cur:
            return 1
        else:
            return 2
        
    def motion_score(self,time):
        #if not self.virtue:
        cur_loc=self.last_obj[0]
        cur_speed=self.last_obj[1]
        delta_speed=self.last_obj[5]
        #cur_direction=self.last_obj[2]
        pred_speed=self.cur_speed+delta_speed
        new_loc=self.cur_obj[0]
        motion_change=self.state_compare()
        if not (self.last_obj[3]=="stand" or self.last_obj[3]=="sink"):
            pred_loc=cur_loc+pred_speed*time
            delta_loc=new_loc-pred_loc
            delta_dis=np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
            score=np.exp(-1*delta_dis)
        elif motion_change==2:
            delta_loc=new_loc-cur_loc
            delta_dis=np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
            score = np.exp(-1*delta_dis/time)
        else:
            delta_loc=new_loc-cur_loc
            delta_dis=np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
            score = np.exp(-1*delta_dis)
        return score

    def state_score(self):
        old_state=self.last_obj[2]
        cur_state=self.cur_obj[2]
        state_score=np.exp(-1*np.abs(cur_state-old_state))
        return state_score
        
    def overall_score(self):
        if not self.virtue:
            score=self.weight[0]*self.motion_score+self.weight[1]*self.state_score
            return score
        else:
            score=self.weight[0]*self.motion_score
            return score
        
class treenode():
    def __init__(self, obj):
        self.obj=obj
        self.child=[]
        self.parent=None
        self.ID=str(obj.id)
        self.score=0
        self.depth=0
        
    def add_child(self,child):
        self.child.append(child)
        
    def remove_child(self,child):
        self.child.remove(child)
            
    def update_score(self,score):
        self.score=score

class TrackTree():
    def __init__(self, root):
        self.len_tree=1
        self.tree_dict=[root]
        self.max_tree=1
        self.bottom=[root]
        
    def add_Node(self, Node, parent):
        if self.len_tree==0:
            self.tree_dict.append(Node)
            self.len_tree+=1
            self.max_tree=1
        else:
            self.tree_dict.append(Node)
            self.len_tree+=1
            Node.parent=parent
            Node.depth=parent.depth+1
            parent.child.append(Node)
            if Node.depth>self.max_tree:
                self.max_tree=Node.depth
                self.bottom=[]
                self.bottom.append(Node)
            elif Node.depth==self.max_tree:
                self.bottom.append(Node)
            else:
                pass
        
    def prune_tree(self,Node):
        temp=Node
        self.bottom.remove(Node)
        while(len(temp.parent.child)==1):
            temp_node=temp.parent
            temp_node.remove_child(temp)
            self.tree_dict.remove(temp)
            temp=temp_node
            len_tree-=1
            
    def search_bottom_node(self,node):
        for i in range(len(self.bottom)):
            if self.bottom[i].positions == node.positions and self.bottom[i].timestamp == node.timestamp:
                return i
            else:
                continue
        return -1
        
          
class Tracker():
    def __init__(self):
        self.tracktree_dict={}
        self.match_dict={}
        
    def add_tree(self,root, id):
        new_track=TrackTree(root)
        self.tracktree_dict.update({id:new_track})
    
    def update_tree(self,id,obj_list):
        Tree=self.tracktree_dict[id]
        Bottom_tree=Tree.bottom
        max_score=0
        max_node=None
        max_index=0
        for j in range(len(Bottom_tree)):
            for i in range(len(obj_list)):
                scorer=Swimmer_score(Bottom_tree[j],Node_list[i],[0.5,0.5])
                score=scorer.overall_score()
                Node=treenode(obj_list[i])
                Node.update_score(score)
                if Node.score>max_score:
                    max_node=Node
                    max_score=Node.score
                    max_index=i
            Tree.add_Node(max_node,Bottom_tree[j])
        return max_index
    
    
    def prune_tracker_tree(self,id,threshold):
        Tree=self.tracktree_dict[id]
        Bottom_tree=Tree.bottom
        score_list=[]
        for i in range(len(Bottom_tree)):
            score_list.append(Bottom_tree[i].score)
        for i in range(len(score_list)):
            if score_list[i]<threshold:
                score_list.remove(score_list[i])
                Tree.prune_tree(Bottom_tree[i])
        index_min=np.argmin(score_list)
        Tree.prune_tree(Bottom_tree[index_min])
        index_max=np.argmax(score_list)
        return index_max
        

    def compare_two_tree_prune(self,id_1,id_2,obj):
        tree_1=self.tracktree_dict[id_1]
        tree_2=self.tracktree_dict[id_2]
        bottom_1=tree_1.bottom
        bottom_2=tree_2.bottom
        if len(bottom_1)==1 and len(bottom_2)==1:
            return 0
        elif len(bottom_1)>1 and len(bottom_2)==1:
            index=bottom_1.search_bottom_node(obj)
            tree_1.prune_tree(bottom_1[index])
            #bottom_1.remove(bottom_1[index])
        elif len(bottom_2)>1 and len(bottom_1)==1:
            index=bottom_2.search_bottom_node(obj)
            tree_2.prune_tree(bottom_2[index])
        else:
            index_1=bottom_1.search_bottom_node(obj)
            index_2=bottom_2.search_bottom_node(obj)
            swim_scorer_1=Swimmer_score(bottom_1[index_1].parent,bottom_1[index_1],[0.5,0.5])
            swim_scorer_2=Swimmer_score(bottom_2[index_2].parent,bottom_2[index_2],[0.5,0.5])
            score_1=swim_scorer_1.overall_score()
            score_2=swim_scorer_2.overall_score()
            if score_1>score_2:
                tree_2.prune_tree(bottom_2[index])
            else:
                tree_1.prune_tree(bottom_1[index])

def naive_scanning_scheme(start_angle, end_angle, skipping):
    if start_angle == -1:
        start=0
    if end_angle == -1:
        end=399
    scanning_range=np.arange(start,end,skipping+1)
    scanning_range.astype(int)
    return scanning_range

def back_forth_scanning_scheme(start_angle, end_angle, step,count):
    if start_angle == -1:
        start_angle = 0
    if end_angle == -1:
        start_angle= 399
    if count%2 ==0:
        scanning_range=np.arange(start_angle, end_angle+1, step)
    else:
        scanning_range = np.arange(end_angle, start_angle-1, -step)
    scanning_range.astype(int)
    return scanning_range


def ignore_scanning_scheme(start_angle, end_angle, scan_continue, skipping):
    scanning_range = []
    current_value = start_angle
    for i in np.arange(start_angle, end_angle + 1, 1):
        current_value = i
        if (i - start_angle) % (scan_continue + skipping) in range(scan_continue):
            scanning_range.append(i)
        else:
            continue
    return scanning_range
    
def tracking_scheme(tracker,obj_list,id_list):
    match_dict=np.zeros(len(id_list))
    for i in range(len(id_list)):
        #for j in range(len(obj_list)):
        _=tracker.update_tree(id_list[i],obj_list)
        max_index=tracker.prune_tracker_tree(id_list[i],0.3)
        match_dict[i]=max_index
    for i in range(len(match_dict)):
        for j in range(i+1,len(match_dict)):
            if match_dict[j]==match_dict[i]:
                tracker.compare_two_tree_prune(id_list[i],id_list[j],obj_list[match_dict[i]])
                

    
    
    
