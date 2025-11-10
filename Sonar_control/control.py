# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:04:08 2020

@author: Haozheng
"""

from brping import Ping360
from brping import definitions
import numpy as np
import time
import matplotlib.pyplot as plt
from sonar_display import show_sonar
from data import *
import torch
from models import *
from skimage.transform import resize
import torchvision.transforms as transforms
import Track_scan as Ts
import argparse

min_human=0.4
max_human=1.5
_firmwareMaxTransmitDuration=500
_firmwareMinTransmitDuration = 5
_samplePeriodSickDuration=25e-9
_firmwareMaxNumberOfPoints = 1200
_firmwareMinSamplePeriod = 80
def calsampleperiod(distance,number_sample):
    return 2*distance/(number_sample*speed_of_sound*_samplePeriodSickDuration)
def adjustTransmitDuration(distance,sample_period):
    transmit_duration=round(8000*distance/speed_of_sound)
    transmit_duration = max(2.5*sample_period/1000, transmit_duration)
    return max(_firmwareMinTransmitDuration, min(transmitDurationMax(sample_period), transmit_duration))

def transmitDurationMax(sample_period):
    return min(_firmwareMaxTransmitDuration, samplePeriod(sample_period) * 64e6)

def samplePeriod(sample_period):
    return sample_period*_samplePeriodSickDuration

def rescan(former_object, distance, number_sample, k, images):
    # change the setting to suit different object
    temp_distance = max(former_object[4], 8)
    temp_number_sample = round(number_sample * temp_distance/distance)
    temp_sample_period = calsampleperiod(temp_distance, temp_number_sample)
    temp_sample_period = round(temp_sample_period)
    temp_transmit_duration = adjustTransmitDuration(temp_distance, temp_sample_period)

    p.set_sample_period(temp_sample_period)
    p.set_number_of_samples(temp_number_sample)
    p.set_transmit_duration(temp_transmit_duration)

    for l in range(k):
        image = np.zeros((number_sample, former_object[2]-former_object[1]+1))
        for i in range(former_object[1], former_object[2]+1):
            p.control_transducer(
                0,
                p._gain_setting,
                i,
                p._transmit_duration,
                p._sample_period,
                p._transmit_frequency,
                p._number_of_samples,
                1,
                0
            )
            p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
            data = [int(n) for n in p._data]
            image[:temp_number_sample, i-former_object[1]] = data
        images = np.concatenate((images, image[:,:, np.newaxis]), axis=2)
    #return images[int(number_sample*former_object[3]/distance):int(number_sample*former_object[4]/distance), :, :]
    return images
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--data', action="store", required=False, type=int, default=1, help="We use fake data or sonar data")
    parser.add_argument('--mode', action="store", required=False, type=int, default=2, help="0-scan one sector, 1-scan one direction, 2-a   ")
    parser.add_argument('--background', action="store", required=False, type=int, default=1,help="Use background substration or close substration")
    parser.add_argument('--start', required=False, type=int, default=0,help="start angle")
    parser.add_argument('--end', required=False, type=int, default=399,help="start angle")
    parser.add_argument('--skip', required=False, type=int, default=0,help="start angle")
    args = parser.parse_args()
    threshold = [10, 200, 3, 100] #object filter
    start_angle = args.start
    stop_angle = args.end
    scan_step = args.skip #only for mode_0
    repeat = 1 # only for mode_1
    fast_scan = 3 # mode_2
    slow_scan = 1 # mode_2
    num_rescan = 3 # mode_2
    reference = "mode_0/2021-09-16-10-06-11.txt"

    if args.data == 1:
        device='COM8'
        baudrate=115200
        p = Ping360()
        p.connect_serial(device, baudrate)
        p.initialize()
        # setting
        speed_of_sound = 1500
        number_sample = 500
        frequency = 750
        distance = 20
        gain_setting = 0

        sample_period = calsampleperiod(distance, number_sample)
        sample_period = round(sample_period)
        transmit_duration = adjustTransmitDuration(distance, sample_period)

        print(sample_period, transmit_duration)
        p.set_gain_setting(gain_setting)
        p.set_transmit_frequency(frequency)
        p.set_sample_period(sample_period)
        p.set_number_of_samples(number_sample)
        p.set_transmit_duration(transmit_duration)
    if args.mode == 0:
        # scan sector
        if args.data == 0:
            number_sample=500
            distance=10
        # adjust the start and end angle
        sonar_img=np.zeros((number_sample,int(400/scan_step)+1))
        fig=plt.figure(figsize=(8,6), dpi = 200)
        local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        fileObject = open("mode_0/"+local_time+'.txt', 'w')
        for x in range(start_angle,stop_angle,scan_step):
            if args.data:
                p.control_transducer(
                    0,  # reserved
                    p._gain_setting,
                    x,
                    p._transmit_duration,
                    p._sample_period,
                    p._transmit_frequency,
                    p._number_of_samples,
                    1,
                    0
                )
                p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                new_message = [int(j) for j in p._data]
            else:
                # fake data
                new_message = np.random.random((number_sample))*1
            fileObject.write(str(x)+" ")
            for j in range(len(new_message)):
                fileObject.write(str(new_message[j])+" ")
            fileObject.write("\n")
            if len(new_message)>0:
                sonar_img[:,int(x/scan_step)]=new_message
        fileObject.close()
        show_sonar(sonar_img, distance)
        plt.savefig("mode_0/"+local_time+".png",dpi=200,bbox_inches = 'tight')
        plt.axis('equal')
        plt.show()
    elif args.mode == 1:
    # continously scan smaller sector
        if args.data == 0:
            number_sample=666
        for r in range(repeat):
            local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            fileObject = open("mode_1/" + local_time + '.txt', 'w')
            t_start = time.time()
            for i in range(start_angle, stop_angle, 1):
                if args.data:
                    p.control_transducer(
                        0,  # reserved
                        p._gain_setting,
                        i,
                        p._transmit_duration,
                        p._sample_period,
                        p._transmit_frequency,
                        p._number_of_samples,
                        1,
                        0
                    )
                    p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                    new_message=[int(j) for j in p._data]
                else:
                    new_message = np.random.random((number_sample))*0.5
                    new_message[30]+=np.sin(2*np.pi*2*time.time())+1
                    new_message[77]+=(np.sin(2*np.pi*2*time.time())+1)*0.6
                    new_message[175]+=(np.sin(2*np.pi*2*time.time())+1)*0.5
                    new_message[277]+=(np.sin(2*np.pi*2*time.time())+1)*0.6
                    plt.pause(0.1)
                fileObject.write(str(i)+" ")
                for j in range(len(new_message)):
                    fileObject.write(str(new_message[j])+" ")
                fileObject.write("\n")
            fileObject.close()
            print((stop_angle-start_angle)/(time.time()-t_start))
    elif args.mode == 2:
    # real deployment
        # load reference data from mode_0
        if args.background == 1:
            sonar_image_ref = np.zeros((number_sample, 400))
            f = open(reference, "r")
            lines = f.readlines()
            for line in lines:
                angle, data = readline(line)
                if len(data) == number_sample:
                    sonar_image_ref[:, angle] = data
            f.close()
        # DNN
        # device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # device = torch.device('cpu')
        # net = LeNet().to(device)
        # net.load_state_dict(torch.load("checkpoint/94.73.pkl"))
        # Norm = transforms.Normalize((41.153403795248266, 41.10783403697201, 41.126265286197004),(16.70487376238919, 16.735242169921765, 16.75909671974799))
        # test_transform = transforms.Compose([transforms.ToTensor(), Norm])
        # sonar control
        angle = start_angle
        angle_former = (start_angle-1)%400
        object_former = []
        object_record = {}
        peaks_record = [[[], []]] * 400
        sonar_img = np.zeros((number_sample, 400))
        local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        fileObject = open("mode_2/" + local_time + '.txt', 'w')
        t_start = time.time()
        while(1):
            if (angle > stop_angle):
                fileObject.close()
                angle = angle - stop_angle - 1 + start_angle
                object_record = {}
                peaks_record = [[]] * 400
                local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                fileObject = open("mode_2/" + local_time + '.txt', 'w')
                print(time.time()-t_start)
                #show_sonar(sonar_img, distance)
                #plt.show()
            p.control_transducer(
                0,  # reserved
                p._gain_setting,
                angle,
                p._transmit_duration,
                p._sample_period,
                p._transmit_frequency,
                p._number_of_samples,
                1,
                0
            )
            p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
            data = [int(j) for j in p._data]
            if len(data) == 0:
                # wrong data
                angle = angle + 1
            else:
                # record data
                fileObject.write(str(angle) + " ")
                for j in range(len(data)):
                    fileObject.write(str(data[j]) + " ")
                fileObject.write("\n")
                if args.background == 1:
                    data = abs(data - sonar_image_ref[:, angle])
                else:
                    data[:round(number_sample/distance)] = 0

                len_sample = distance / number_sample
                data_filter = smooth(data, len_sample, 0)
                local_var = smooth(abs(data - data_filter), len_sample, 1)
                peaks, dict = detect(data_filter, len_sample, local_var)
                new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                sonar_img[:, angle] = data
                rmax = 0
                angle_add = fast_scan
                for o in object_former:
                    if o[1] not in overlap:
                        if filter(object_former[o], threshold):
                            if object_former[o][4] > rmax:
                                rmax = object_former[o][4]
                                r = object_former[o]
                    else:
                        angle_add = slow_scan
                if rmax!=0:
                    print(r)
                    images = sonar_img[:, r[1]: r[2] + 1, np.newaxis]
                    images = rescan(r, distance, number_sample, num_rescan, images)
                    info = str(r[1]) + '-' + str(r[2]) + '-' + str(int(r[3]/len_sample)) + '-' + str(int(r[4]/len_sample)) + '_'
                    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    np.save("mode_2/" + info + local_time + '.npy', images)
                    # images = test_transform(resize(images, (51, 11, 3)))
                    # return to former setting
                    sample_period = calsampleperiod(distance, number_sample)
                    sample_period = round(sample_period)
                    transmit_duration = adjustTransmitDuration(distance, sample_period)
                    p.set_sample_period(sample_period)
                    p.set_number_of_samples(number_sample)
                    p.set_transmit_duration(transmit_duration)
                    # do classification
                    # with torch.no_grad():
                    #     images = torch.unsqueeze(images, 0)
                    #     output = net(images.to(device, dtype=torch.float))
                    #     print(output.data)
                angle_former = angle
                object_former = new_object
                for i in range(1, angle_add):
                    sonar_img[:, (angle + i) % 400] = data
                angle = angle + angle_add
    elif args.mode == 3:
        scanning_angle=Ts.naive_scanning_scheme(start_angle,stop_angle,scan_step)
        #sonar_img=np.zeros((number_sample,int(400/scan_step)+1))
        #fig=plt.figure(figsize=(8,6), dpi = 200)
        local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        fileObject = open("mode_3/"+local_time+'.txt', 'w')
        for i in range(len(scanning_angle)):
            if args.data:
                p.control_transducer(
                    0,  # reserved
                    p._gain_setting,
                    scanning_angle[i],
                    p._transmit_duration,
                    p._sample_period,
                    p._transmit_frequency,
                    p._number_of_samples,
                    1,
                    0
                )
                p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                new_message = [int(j) for j in p._data]
            else:
                # fake data
                new_message = np.random.random((number_sample))*1
            fileObject.write(str(scanning_angle[i])+" ")
            for j in range(len(new_message)):
                fileObject.write(str(new_message[j])+" ")
            fileObject.write("\n")
            #if len(new_message)>0:
            #    sonar_img[:,int(x/scan_step)]=new_message
        fileObject.close()
        #show_sonar(sonar_img, distance)
        #plt.savefig("mode_0/"+local_time+".png",dpi=200,bbox_inches = 'tight')
        #plt.axis('equal')
        #plt.show()
    elif args.mode==4:
        scanning_angle=[]    #Ts.track_scanning_scheme(start_angle,stop_angle,scan_step)
        #sonar_img=np.zeros((number_sample,int(400/scan_step)+1))
        #fig=plt.figure(figsize=(8,6), dpi = 200)
        local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        fileObject = open("mode_3/"+local_time+'.txt', 'w')
        for i in range(len(scanning_angle)):
            if args.data:
                p.control_transducer(
                    0,  # reserved
                    p._gain_setting,
                    scanning_angle[i],
                    p._transmit_duration,
                    p._sample_period,
                    p._transmit_frequency,
                    p._number_of_samples,
                    1,
                    0
                )
                p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                new_message = [int(j) for j in p._data]
            else:
                # fake data
                new_message = np.random.random((number_sample))*1
            fileObject.write(str(scanning_angle[i])+" ")
            for j in range(len(new_message)):
                fileObject.write(str(new_message[j])+" ")
            fileObject.write("\n")
            #if len(new_message)>0:
            #    sonar_img[:,int(x/scan_step)]=new_message
        fileObject.close()
        
    else:
        t_start = time.time()
        for x in range(0, 10, 1):
            p.control_transducer(
                0,  # reserved
                p._gain_setting,
                x,
                p._transmit_duration,
                p._sample_period,
                p._transmit_frequency,
                p._number_of_samples,
                1,
                0
            )
            p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
            data = [int(j) for j in p._data]
            print(len(data))
        print(time.time() - t_start)

