# Mobicom2025 Artifact Evaluation: AquaScan: A Sonar-based Underwater Sensing System for Human Activity Monitoring

## This repo contains the code, deployment instructions, Experiment instruction and detail usage of each function.

## Installation and Preparation for Artifact evaluation

### Hardware depandencies

Computing platform: A server with CPU and GPU. Our code is tested on a computer with 7950x3D CPU, 4090 24GB GPU and 64GB RAM.

Sensor Node:
* Control Unit: Raspberry Pi 4B model 8GB RAM
* Ping360 Scanning Sonar: https://bluerobotics.com/store/sonars/imaging-sonars/ping360-sonar-r1-rp/


### Software dependencies
For common python packages such as numpy, matplotlib, you can download through pip or conda install these packages

Some important package version:
* numpy version: 1.24.3
* scikit-image: 0.20.0
* scikit-learn: 1.3.0
* matplotlib: 3.7.1
* PyTorch: 2.1.0
* CUDA Version: 12.1
* Opencv-python: 4.8.1.78

If you want to control scanning sonar (Ping 360) to collect data by yourselves, you should install the brping packages through run the command below:
```bash
pip install --user bluerobotics-ping --upgrade
```
### Deployment Instruction
To deploy Ping360 Sonar in the pool, we use a stand shown in the figure below to fix the position of sonars.

![image](https://anonymous.4open.science/r/AquaScan_Artifact-0E81/figure/setup-2-2.jpg)

### Dataset Setup
Download the dataset from the onedrive link. 

Unzip the file and put raw_data, raw_labels, scripts, checkpoints under the path: ./AquaScan_Artifact/image_reconstruction/AquaScan_data

Run the script ./AquaScan_Artifact/image_reconstruction/prepare.sh
```bash
cd image_reconstruction
bash prepare.sh
```
## Experiments Instruction
### Before Experiments
Please read the README.md in AquaScan_data to learn about the data and the label. Also, it includes the description of scripts.

### Image Reconstruction
Run the script 0229_recover.sh, 0807_recover.sh, 0814_recover.sh, 0821_recover.sh in the folder image_reconstruction

These four script will process unreconstructed dataset in the raw_data.

```bash
cd image_reconstruction
bash 0229_recover.sh;bash 0807_recover.sh;bash 0814_recover.sh;bash 0821_recover.sh
```

### Object Detection & Tracking & Moving Detection

Run the script label_0229.sh,label_0807.sh, label_0814.sh,label_0821.sh

For each script label_xxxx.sh, it contains the command to run the pre_sonar_bias.py pre_sonar.py/pre_sonar_opt_yoho.py label2dis.py track.py and moving_detect.py with pre-defined parameters.

The script will generate visualization in the file with name xxxx_localize_3, intermediate results of tracking track_xxxx and result_xxxx. 

```bash
cd ..
bash label_0229.sh;bash label_0807.sh;bash label_0814.sh;bash label_0821.sh
```

### Activity Recognition
Run the script infe_0229.sh,infe_0807.sh, infe_0814.sh,infe_0821.sh
```bash
bash create_folder.sh
bash infe_0229.sh;bash infe_0807.sh;bash infe_0814.sh;bash infe_0821.sh
```

### Plot confusion matrix and show the numercial results
```bash
python cal_results.py
```

### Remove the produced results.

```bash
bash remove_history_0229.sh;bash remove_history_0807.sh;bash remove_history_0814.sh;bash remove_history_0821.sh
bash remove_results.sh
cd image_reconstruction
bash remove_recover.sh
```

### Expected Outputs:
Object detection results:

* F1_score: 0.861731843575419
* miss_rate: 0.060167555217060166
* IoU_aver: 0.5205157489569353

Accuracy: 0.916972814107274

## Detail of each function

### Sonar Control
Please see the README.md in Sonar_control folder.

### image reconstruction
Please see the README.md in image_reconstruction folder.

### object detection
We have three code:
* pre_sonar_bias.py: remove the bias caused by mechnical rotation of sonar (seldom happen)
* pre_sonar.py: denoise sonar images and detect object on the sonar image.
* pre_sonar_opt_yoho: optimze the dynamic object detection with binary search


