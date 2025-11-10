import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import subprocess


def split_datalist(datalist_dir, save_dir, k_fold=5, random_state=42):
    '''
    Split datalist into k_fold parts, and save them into save_dir.
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    datalist = [line.split() for line in open(datalist_dir).readlines()]
    data_paths = [info[0] for info in datalist]
    label_paths = [info[1] for info in datalist]
    labels = [str(open(label_path).readlines()[1].strip()) for label_path in label_paths]
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    for i, (train_index, val_index) in enumerate(skf.split(data_paths, labels)):
        train_list = os.path.join(save_dir, "train_{}.txt".format(i))
        val_list = os.path.join(save_dir, "val_{}.txt".format(i))
        with open(train_list, "w") as f:
            for idx in train_index:
                f.write("{} {}\n".format(data_paths[idx], label_paths[idx]))
        
        with open(val_list, "w") as f:
            for idx in val_index:
                f.write("{} {}\n".format(data_paths[idx], label_paths[idx]))
        print("Save train list to {}".format(train_list))
        print("Save val list to {}".format(val_list))
        
def split_datalist_val(datalist_dir, aug_dir, save_dir,aug_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    datalist = [line.split() for line in open(datalist_dir).readlines()]
    #auglist=[line.split() for line in open(aug_dir).readlines()]
    if aug!="none":
        auglist=[line.split() for line in open(aug_dir).readlines()]
        aug_data_list=[info[0] for info in auglist]
        aug_label_list=[info[1] for info in auglist]
    data_paths = [info[0] for info in datalist]
    label_paths = [info[1] for info in datalist]
    labels = [str(open(label_path).readlines()[1].strip()) for label_path in label_paths]
    data_train,data_test,label_train,label_test=train_test_split(data_paths,label_paths,test_size=0.8,random_state=42)
    data_test_train,data_test_val,label_test_train,label_test_val=train_test_split(data_test,label_test,test_size=0.5,random_state=42)
    
    train_path= os.path.join(save_dir, "train_small_"+aug_name+".txt")
    test_path= os.path.join(save_dir, "test_small_"+aug_name+".txt")
    val_path=os.path.join(save_dir, "val_small_"+aug_name+".txt")
    
    with open(train_path,"w") as f:
        for idx in range(len(data_train)):
            f.write("{} {}\n".format(data_paths[idx], label_paths[idx]))
        if aug!="none":
            for idx in range(len(aug_data_list)):
                f.write("{} {}\n".format(aug_data_list[idx], aug_label_list[idx]))
    f.close()
    
    with open(test_path,"w") as f:
        for idx in range(len(data_test_train)):
            f.write("{} {}\n".format(data_test_train[idx], label_test_train[idx]))
    f.close()
    
    with open(val_path,"w") as f:
        for idx in range(len(data_test_val)):
            f.write("{} {}\n".format(data_test_val[idx], label_test_val[idx]))
    f.close()
    
def split_datalist_save(datalist_dir,save_dir,save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    datalist = [line.split() for line in open(datalist_dir).readlines()]
    #auglist=[line.split() for line in open(aug_dir).readlines()]
    data_paths = [info[0] for info in datalist]
    label_paths = [info[1] for info in datalist]
    #labels = [str(open(label_path).readlines()[1].strip()) for label_path in label_paths]

    save_path= os.path.join(save_dir, save_name+".txt")
    
    with open(save_path,"w") as f:
        for idx in range(len(data_paths)):
            f.write("{} {}\n".format(data_paths[idx], label_paths[idx]))
    f.close()

def split_datalist_save_skip(datalist_dir,save_dir,save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    datalist = [line.split() for line in open(datalist_dir).readlines()]
    #auglist=[line.split() for line in open(aug_dir).readlines()]
    data_paths = [info[0] for info in datalist]
    label_paths = [info[1] for info in datalist]
    skip_paths = [info[2] for info in datalist]
    #labels = [str(open(label_path).readlines()[1].strip()) for label_path in label_paths]

    save_path= os.path.join(save_dir, save_name+".txt")
    
    with open(save_path,"w") as f:
        for idx in range(len(data_paths)):
            f.write("{} {} {}\n".format(data_paths[idx], label_paths[idx], skip_paths[idx]))
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist_dir", type=str, required=True, help="Path to datalist.txt")
    parser.add_argument("--aug_dir", type=str, required=False, help="Aug Path to datalist.txt")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save datalist")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--aug_type", type=str, default="none")
    parser.add_argument("--save",type=int, default=0)
    parser.add_argument("--save_name",type=str, required=False, default="data")
    args = parser.parse_args()
    #split_datalist(args.datalist_dir, args.save_dir, args.k_fold, args.random_state)
    aug = args.aug_type
    datalist_dir = args.datalist_dir
    save_dir = args.save_dir
    #for i in range(len(aug)):
    datalist_dir_aug = args.aug_dir
    save_name=args.save_name
    if args.save==0:
        split_datalist_val(datalist_dir,datalist_dir_aug,save_dir,aug)
    else:
        split_datalist_save(datalist_dir,save_dir,save_name)
