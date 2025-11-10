'''
analyze the response data
'''
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import glob

import shutil
min_human=0.4
max_human=1.5
def smooth(data, len_sample, k):
    if k == 0:
        len_filter=round(min_human/len_sample)
    else:
        len_filter=round(max_human/len_sample)
    return np.convolve(data, np.ones(len_filter)/len_filter, mode="same")
def detect(data, len_sample, threshold):
    max_peak = round(max_human / len_sample)
    min_peak = round(min_human / len_sample)
    peaks, dict = find_peaks(data,distance = min_peak, width = [min_peak,max_peak],prominence = 3 *threshold)
    return peaks, dict
def compensate(data):
    # compensate the further and closer response
    if len(np.shape(data))==2:
        d1, d2 = np.shape(data)
        weight = np.arange(0, 1, 1 / d1)
        weights = np.tile(weight[:, np.newaxis], (1, d2))
    else:
        d1 = len(data)
        weights = np.arange(0, 1, 1 / d1)
    return data*weights
def readline(line):
    line = line.split()
    line = list(map(int, line))
    angle = line[0]
    data = line[1:]
    return angle, data
def update_record(peaks_record, object_record, dict, angle, a, len_sample):
    new_object = {}
    overlap = {}
    if len(dict['left_ips']):
        peaks_record[angle] = np.array([dict['left_ips'], dict['right_ips']])
        J = len(peaks_record[angle][1])
        if len(peaks_record[a]) > 0:
            K = len(peaks_record[a][1])
        else:
            K = 0
        if len(peaks_record[a]) != 0 and len(peaks_record[angle]) != 0:
            ratio_mat = np.zeros((J, K))
            for j in range(J):
                 for k in range(K):
                    ratio = (min(peaks_record[a][1][k], peaks_record[angle][1][j]) - max(peaks_record[a][0][k], peaks_record[angle][0][j]))/\
                            (max(peaks_record[a][1][k], peaks_record[angle][1][j]) - min(peaks_record[a][0][k], peaks_record[angle][0][j]))
                    ratio_mat[j, k] = ratio
            for j in range(J):
                if K > 0:
                    if max(ratio_mat[j, :]) > 0.5:
                        kmax = np.argmax(ratio_mat[j, :])
                        if kmax not in overlap:
                            overlap[kmax] = [j]
                        else:
                            overlap[kmax].append(j)
                    else:
                        object_record[angle, j] = [len_sample ** 2 * (peaks_record[angle][1][j] ** 2 - peaks_record[angle][0][j] ** 2) * np.pi / 200, \
                                                   angle, angle, len_sample * peaks_record[angle][0][j],len_sample * peaks_record[angle][1][j], dict['prominences'][j]]
                        new_object[angle, j] = object_record[angle, j]
                else:
                    object_record[angle, j] = [len_sample ** 2 * (peaks_record[angle][1][j] ** 2 - peaks_record[angle][0][j] ** 2) * np.pi / 200, \
                                               angle, angle, len_sample * peaks_record[angle][0][j],len_sample * peaks_record[angle][1][j], dict['prominences'][j]]
                    new_object[angle, j] = object_record[angle, j]
            for k in overlap.keys():
                js = overlap[k]
                for j in js:
                    object_record[angle, j] = object_record[a, k]
                    size_angle = (object_record[angle, j][2] - object_record[angle, j][1]) + 1
                    object_record[angle, j][0] += len_sample**2 * (peaks_record[angle][1][j]**2 - peaks_record[angle][0][j]**2) * np.pi / 200
                    object_record[angle, j][5] = dict['prominences'][j] * abs(angle-a) / (size_angle + abs(angle-a))  + \
                                                 object_record[angle, j][5] * size_angle / (size_angle + abs(angle-a))
                    object_record[angle, j][1] = min(object_record[angle, j][1], angle)
                    object_record[angle, j][2] = max(object_record[angle, j][2], angle)
                    object_record[angle, j][3] = len_sample*peaks_record[angle][0][j] * abs(angle-a) / (size_angle + abs(angle-a))  + \
                                                 object_record[angle, j][3] * size_angle / (size_angle + abs(angle-a))
                    object_record[angle, j][4] = len_sample*peaks_record[angle][1][j]  * abs(angle - a) / (size_angle + abs(angle - a)) + \
                                                 object_record[angle, j][4] * size_angle / (size_angle + abs(angle - a))
                    new_object[angle, j] = object_record[angle, j]
                object_record.pop((a,k))
        elif len(peaks_record[a]) == 0 and len(peaks_record[angle]) != 0:
            for j in range(len(peaks_record[angle][1])):
                object_record[angle, j] = [len_sample**2 * (peaks_record[angle][1][j]**2 - peaks_record[angle][0][j]**2) * np.pi / 200, \
                                        angle, angle, len_sample * peaks_record[angle][0][j], len_sample * peaks_record[angle][1][j], dict['prominences'][j]]
                new_object[angle, j] = object_record[angle, j]
    return new_object, overlap
def filter(r, threshold):
    if r[0]*r[5] >= threshold[0] and r[0]*r[5]<=threshold[1] and r[0] <= threshold[2] and r[5] <= threshold[3]:
        return True
    else:
        return False
def compare(g, results, threshold):
    if len(results)==0:
        return 100, []
    else:
        c = 100 * np.ones((len(results)))
        keys = list(results.keys())
        for i in range(len(keys)):
            result = results[keys[i]]
            if filter(result, threshold):
                c[i] = abs((g[0] - (result[4] + result[3])/2))/20 + abs((g[1] - (result[2] + result[1])/2))/100
        return min(c), results[keys[np.argmin(c)]]
def detect_object_lines(lines, limit, threshold):
    a = {}
    scan_range = 20
    angle_former = 0
    object_former = {}
    object_record = {}
    peaks_record = [[[], []]] * 400
    for i in range(len(lines)):
        angle, data = readline(lines[i])
        if len(data) == 0:
            continue
        len_sample = scan_range / len(data)
        data_filter = smooth(data, len_sample, 0)
        local_var = smooth(abs(data - data_filter), len_sample, 1)
        peaks, dict = detect(data_filter, len_sample, local_var)
        new_object, overlap, object_record = update_record(peaks_record, object_record, dict, angle, angle_former,len_sample)
        rmax = 0
        for o in object_former:
            if o[1] not in overlap:
                if filter(object_former[o], threshold):
                    if object_former[o][4] > rmax:
                        rmax = object_former[o][4]
                        r = object_former[o]
        if rmax != 0:
            if r[4] < limit:
                a[angle] = r
        object_former = new_object
        angle_former = angle
    return a
def detect_object_mat(mat, threshold):
    file_object = []
    object_info = {}
    num_sample = 500
    scan_range = 20
    angle_former = 399
    object_former = {}
    object_record = {}
    peaks_record = [[[], []]] * 400
    for angle in range(mat.shape[1]):
        data = mat[:, angle]
        len_sample = scan_range / num_sample
        data_filter = smooth(data, len_sample, 0)
        local_var = smooth(abs(data - data_filter), len_sample, 1)
        peaks, dict = detect(data_filter, len_sample, local_var)
        new_object, overlap, object_record = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
        rmax = 0
        for o in object_former:
            if o[1] not in overlap:
                if filter(object_former[o], threshold):
                    if object_former[o][4] > rmax:
                        rmax = object_former[o][4]
                        r = object_former[o]
        if rmax != 0:
            file_object.append(file)
            object_info[file] = r
        object_former = new_object
        angle_former = angle
    return file_object, object_info
def time2label1(t, Action, Submerge):
    for i in range(len(Submerge)):
        if t < Submerge[i][1]:
            break
    d = i % 2
    for s in Submerge:
        if t >= s[0] and t <= s[1]:
            return 2, d
    for a in Action:
        if t >= a[0] and t <= a[1]:
            return 1, d
    return 0, d
def time2label2(d, r):
    r_mean = (int(r[0]) + int(r[1]))/2
    d_mean = (int(r[2]) + int(r[3]))/2
    if d >= 0 and d <=3: # drown-swim
        if r_mean > 210 and d_mean > 200:
            return 1
        else:
            return 3
    elif d == 4:
        if r_mean > 200 and d_mean < 150:
            return 1
        else:
            return 3
    elif d == 5:
        if r_mean > 200 and d_mean > 159:
            return 1
        else:
            return 3
    elif d >= 6 and d<=8: # drown-still
        if r_mean > 210:
            return 0
        else:
            return 1
    elif d == 9: # both drown
        return 1
    elif d == 10: # overlap-1
        if d_mean < 75:
            return 0
        else:
            return 3
    elif d == 11: # overlap-2
        if d_mean < 75:
            return 1
        else:
            return 0
    elif d == 12: # overlap-3
        if d_mean > 75:
            return 1
        else:
            return 0
    elif d == 13: # noise pattern
        return 0
    else: # four people
        return 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='analyse experiment data')
    parser.add_argument('--exp', action="store", required=False, type=int, default=7, help="We have collected multiple dataset")
    args = parser.parse_args()
    p_threshold = 0.1
    threshold = [30, 150, 3, 100]
    if args.exp == 1:
        # for the first experiment
        ref_20 = "first_dataset/20_big/20m_ref.txt"
        ref_10 = "first_dataset/10_big/10m_ref.txt"
        dir_20 = "first_dataset/20_small/"
        dir_10 = "first_dataset/10_small/"
        ref = ref_10
        rootdir = dir_10
        f = open(ref,"r")
        num_sample = 500
        scan_range = 10
        sonar_image_ref = np.zeros((num_sample, 400))
        lines = f.readlines()
        for line in lines:
            angle, data = readline(line)
            if len(data) == num_sample:
                sonar_image_ref[:,angle] = data
        f.close()

        files = glob.glob(rootdir+'*.txt')
        a = []
        correct = 0
        whole = 0
        i = 0
        for file in files:
            f = open(file, "r")
            lines = f.readlines()
            #lines = lines[::2]
            data_buffer = np.zeros((num_sample, len(lines)))
            angle_former = 0
            object_record = {}
            peaks_record = [[]] * 400
            j = 0
            plt.figure(1)
            for line in lines:
                if j>=1:
                    angle_former = angle
                angle, data = readline(line)
                if angle < angle_former:
                    whole += 1
                    for k in list(object_record.keys()):
                        if object_record[k][0] <= 0.4:
                            del object_record[k]
                        elif object_record[k][3]>=6 and object_record[k][4]<=8:
                            correct+=1
                            break
                    print(object_record)
                    object_record = {}
                    angle_former = 0
                    peaks_record = [[]] * 400
                data = abs(data-sonar_image_ref[:, angle])
                # whether compensate attenuation
                # data = compensate(data)
                len_sample = scan_range / len(data)
                data_filter = smooth(data, len_sample, 0)
                local_var = smooth(abs(data - data_filter), len_sample, 1)
                peaks, dict = detect(data_filter, len_sample, local_var)
                update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                #plt.plot(peaks, data_filter[peaks], linestyle='None', marker='o', markersize=10)
                #plt.plot(data_filter)
                data_buffer[:, j] = data_filter
                j = j + 1
                # 233 for 20 meters, 240 for 10 meters
                if angle == 240:
                    a.append(data_filter)
            plt.xlabel('distance')
            plt.ylabel('response')
            plt.title('scan result from multiple beams')
            f.close()
            #plt.pause(0.01)
            #plt.clf()
            #plt.figure(2)
            freq = np.fft.fftfreq(len(lines), d=1/11)
            freq = np.fft.fftshift(freq)
            freq = np.round(freq, 1)
            #temporal_info(data_buffer, freq, vertical_s=round(max_human/len_sample/2), bins_freq=3, num_figure=3, mode=0)
            #plt.pause(0.01)
            #plt.clf()
        plt.figure(3)
        plt.plot(np.sqrt(np.var(a,0))/np.mean(a,0))
        plt.show()
        print(correct/whole)
    elif args.exp == 2:
        # remake storage folder
        shutil.rmtree('2/images/')
        os.mkdir('2/images/')
        reference = "2/reference.txt"
        f = open(reference, "r")
        num_sample = 500
        scan_range = 20
        sonar_image_ref = np.zeros((num_sample, 400))
        detection = True
        lines = f.readlines()
        for line in lines:
            angle, data = readline(line)
            if len(data) == num_sample:
                sonar_image_ref[:, angle] = data
        f.close()
        color = {'5/':'b', '10/': 'r', '15/':'y'}
        o = 0
        for dis in ['15/']:
            if dis == '5/':
                if detection:
                    directories = ['static1', 'static3', 'Noaction1A', 'Noaction2A', 'Noaction2B','action1A', 'action1B', 'action2A']
                else:
                    directories = ['Noaction1A', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A']
                ground_truths = {'static1': [(5.5,170),(4.2, 203)],'static3':[(5,168),(4.5,204)],'Noaction1A':[(5.2,160)],'Noaction2A':[(5.2,157)],'Noaction2B':[(5,200)],'action1A':
                                 [(5.2,160)],'action1B':[(5, 195)],'action2A':[(6, 165)]}
            elif dis == '10/':
                if detection:
                    directories = ['static1', 'static3', 'Noaction1B', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A', 'action2B']
                else:
                    directories = ['Noaction1B', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A', 'action2B']
                ground_truths = {'static1': [(10, 178), (10, 196)], 'static3': [(9.5, 175), (9.3, 195)],'Noaction1B': [(8.5, 200)], 'Noaction2A': [(7.5, 175)],
                                 'Noaction2B': [(8, 196)], 'action1A':[(7.8, 175)], 'action1B': [(7.8, 192)], 'action2A': [(7, 170)], 'action2B':[(7.2, 190)]}
            else:
                if detection:
                    directories = ['static1', 'static3', 'Noaction1A', 'action1A', 'action2A']
                else:
                    directories = ['Noaction1A', 'action1A', 'action2A']
                ground_truths = {'static1': [(16, 185)], 'static3': [(16, 185)],'Noaction1A': [(16.7, 185)], 'action1A': [(17, 184)], 'action2A': [(17, 186)]}
            directories = directories
            for d in directories:
                path = '2/' + dis + d +'/*.txt'
                files = glob.glob(path)
                ground_truth = ground_truths[d]
                correct = 0
                size = []
                intensity = []
                for file in files:
                    f = open(file, 'r')
                    lines = f.readlines()
                    sonar_img = np.zeros((num_sample, len(lines)))
                    angle_former = 0
                    object_record = {}
                    peaks_record = [[[],[]]] * 400
                    for i in range(len(lines)):
                        if i >= 1:
                            angle_former = angle
                        angle, data = readline(lines[i])
                        data = abs(data - sonar_image_ref[:, angle])
                        len_sample = scan_range / len(data)
                        data_filter = smooth(data, len_sample, 0)
                        local_var = smooth(abs(data - data_filter), len_sample, 1)
                        peaks, dict = detect(data_filter, len_sample, local_var)
                        new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                        sonar_img[:, i] = data_filter
                        if i == 0:
                            first_angle = angle
                        elif i == len(lines)-1:
                            last_angle = angle
                    f.close()
                    for g in ground_truth:
                        c, r = compare(g, object_record,  threshold)
                        if c <= p_threshold:
                            correct = correct + 1
                            crop_image = sonar_img[int(r[3]/len_sample):int(r[4]/len_sample)+1,r[1]-first_angle:r[2]-first_angle+1]
                            np.save('2/' + 'images/' + d + '-' + str(o) + '.npy', crop_image)
                            o = o + 1
                            #plt.scatter(r[2]-r[1], (r[4]-r[3]), c = 'blue')
                            plt.scatter(r[0], r[5], c='blue')
                        else:
                            #plt.scatter(r[2]-r[1], (r[4]-r[3]), c = 'green')
                            plt.scatter(r[0], r[5], c='green')
                    # plt.imshow(sonar_img, cmap='gray',aspect=len(lines)/num_sample)
                    # plt.yticks([0,49,99,149,199,249,299,349,399,449,499],[0,2,4,6,8,10,12,14,16,18,20])
                    # plt.xticks(np.arange(len(lines))[::int(len(lines)/4)], np.arange(first_angle, last_angle+1)[::int(len(lines)/4)])
                    # plt.show()
                print(dis[:-1], d, correct/len(files)/len(ground_truth))
        plt.show()
    elif args.exp == 3:
        n = 0
        a = 0
        for sonar in ["sonar_1/", "sonar_2/"]:
            shutil.rmtree('3/' + sonar + 'images/')
            os.mkdir('3/' + sonar + 'images/')
            reference = "3/" + sonar + "reference.txt"
            f = open(reference, "r")
            num_sample = 500
            scan_range = 20
            sonar_image_ref = np.zeros((num_sample, 400))
            lines = f.readlines()
            for line in lines:
                angle, data = readline(line)
                if len(data) == num_sample:
                    sonar_image_ref[:, angle] = data
            f.close()
            directories = ['Noaction1', 'Noaction2', 'action1', 'action2']
            #directories = ['action1']
            ground_truths = {"sonar_1/":{'Noaction1':[(6,167),(3.5, 154)], 'Noaction2':[(6,164),(3.8,155)], 'action1':[(6,165),(4,158)], 'action2':[(6.6,165),(4,155)],},
                             "sonar_2/":{'Noaction1':[(3.8,232),(5.2,262)], 'Noaction2':[(3,230),(5,260)], 'action1':[(3,230),(5,260)], 'action2':[(3,220),(4.5,260)]}}
            for d in directories:
                path = '3/' + sonar + d
                files = os.listdir(path)
                ground_truth = ground_truths[sonar][d]
                correct = 0
                size = []
                intensity = []
                for file in files:
                    f = open(path + '/' + file, 'r')
                    lines = f.readlines()
                    sonar_img = np.zeros((num_sample, len(lines)))
                    angle_former = 0
                    object_former = {}
                    object_record = {}
                    peaks_record = [[[],[]]] * 400
                    for i in range(len(lines)):
                        if i >= 1:
                            angle_former = angle
                        angle, data = readline(lines[i])
                        if len(data) == 0:
                            continue
                        data = abs(data - sonar_image_ref[:, angle])
                        len_sample = scan_range / len(data)
                        data_filter = smooth(data, len_sample, 0)
                        local_var = smooth(abs(data - data_filter), len_sample, 1)
                        peaks, dict = detect(data_filter, len_sample, local_var)
                        new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                        sonar_img[:, i] = data_filter
                        object_former = new_object
                        if i == 0:
                            first_angle = angle
                        elif i == len(lines) - 1:
                            last_angle = angle
                    f.close()
                    #print('whole')
                    # for o in object_record:
                    #     if filter(object_record[o], [25, 80, 1.5, 100]):
                    #         a = a + 1
                    #         print(object_record[o])
                    #print('successful')
                    for g in ground_truth:
                        c, r = compare(g, object_record,  threshold)
                        if c <= p_threshold:
                            #print(r)
                            correct = correct + 1
                            crop_image = sonar_img[int(r[3] / len_sample):int(r[4] / len_sample) + 1, r[1] - first_angle:r[2] - first_angle + 1]
                            np.save('3/' + sonar + 'images/' + d + '-' + str(n) + '.npy', crop_image)
                            n = n + 1
                    # for o in object_record:
                    #     temp_object = {}
                    #     r = object_record[o]
                    #     temp_object[o] = r
                    #     for g in ground_truth:
                    #         c, _ = compare(g, temp_object, 25)
                    #         if c <= p_threshold:
                    #             plt.scatter(r[0], r[5], c='b', marker='x', s = 10)
                    #             break
                    #         else:
                    #             plt.scatter(r[0], r[5], c = 'g', marker='o', s = 10)
                    # plt.imshow(sonar_img, cmap='gray',aspect=len(lines)/num_sample)
                    # plt.yticks([0,49,99,149,199,249,299,349,399,449,499],[0,2,4,6,8,10,12,14,16,18,20])
                    # plt.xticks(np.arange(len(lines))[::int(len(lines)/4)], np.arange(first_angle, last_angle+1)[::int(len(lines)/4)])
                    # plt.show()
                print(sonar + d, correct / len(files))
        plt.show()
    elif args.exp == 4:
        n = 0
        a = 0
        num_sample = 500
        scan_range = 20
        sonar_image_refs = {}
        for sonar in ["sonar_1/", "sonar_2/"]:
            for f in os.listdir('4/' + sonar):
                if f.split('.')[-1] == 'txt':
                    f_name = f.split('.')[0]
                    sonar_image_refs[f_name] = np.zeros((num_sample, 400))
                    reference = "4/" + sonar + f
                    f = open(reference, "r")
                    lines = f.readlines()
                    for line in lines:
                        angle, data = readline(line)
                        if len(data) == num_sample:
                            sonar_image_refs[f_name][:, angle] = data
                    f.close()
            directories = ["Normal", 'Action', 'Submerge', 'Random']
            # ground_truths = {"sonar_1/": {"Normal": [(6, 167), (3.5, 154)], 'Action': [(6, 164), (3.8, 155)],'Submerge': [(6, 165), (4, 158)], },
            #                  "sonar_2/": {"Normal": [(3.8, 232), (5.2, 262)], 'Action': [(3, 230), (5, 260)],'Submerge': [(3, 230), (5, 260)], }}
            for d in directories:
                path = '4/' + sonar + d
                files = os.listdir(path)
                correct = 0
                for file in files:
                    if file.split('.')[0] > '2021-5-13-11-25-22':
                        sonar_image_ref = sonar_image_refs['reference2']
                    else:
                        sonar_image_ref = sonar_image_refs['reference1']
                    if file.split('.')[-1] == 'txt':
                        f = open(path + '/' + file, 'r')
                        lines = f.readlines()
                        sonar_img = np.zeros((num_sample, 400))
                        angle_former = 0
                        object_former = {}
                        object_record = {}
                        peaks_record = [[[], []]] * 400
                        for i in range(len(lines)):
                            angle, data = readline(lines[i])
                            if len(data) == 0:
                                continue
                            data = abs(data - sonar_image_ref[:, angle])
                            len_sample = scan_range / len(data)
                            data_filter = smooth(data, len_sample, 0)
                            local_var = smooth(abs(data - data_filter), len_sample, 1)
                            peaks, dict = detect(data_filter, len_sample, local_var)
                            new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former, len_sample)
                            sonar_img[:, angle] = data
                            rmax = 0
                            for o in object_former:
                                if o[1] not in overlap:
                                    if filter(object_former[o], threshold):
                                        if object_former[o][4] > rmax:
                                            rmax = object_former[o][4]
                                            r = object_former[o]
                            if rmax!=0:
                                print(r)
                            object_former = new_object
                            angle_former = angle
                        f.close()
                    else:
                        continue
    elif args.exp == 5:
        num_sample = 500
        scan_range = 20
        sonar_image_refs = {}
        for sonar in ["sonar_1/", "sonar_2/"]:
            shutil.rmtree('5/' + sonar + 'images/')
            os.mkdir('5/' + sonar + 'images/')
            directories = ["Normal", 'Action', 'Submerge', 'Full']
            for d in directories:
                path = '5/' + sonar + d
                files = os.listdir(path)
                object_info = {}
                for file in files:
                    if file.split('.')[-1] == 'txt':
                        f = open(path + '/' + file, 'r')
                        lines = f.readlines()
                        f.close()
                        obj = detect_object_lines(lines, 10, threshold)
                        if obj != {}:
                            object_info[file] = obj
                        else:
                            pass
                keys = list(object_info.keys())
                for i, o in enumerate(keys):
                    m = 0
                    n = 100
                    o1 = o.split('.')[0]
                    o2 = keys[min(i+1, len(keys)-1)].split('.')[0]
                    for angle in object_info[o]:
                        degree = object_info[o][angle][2] - object_info[o][angle][1]
                        if degree > m:
                            m = degree
                            r = object_info[o][angle]
                            deg = int((r[4] - r[3]) * 25)
                    for file in files:
                        if file.split('.')[-1] == 'npy' and file.split('.')[0] >= o1 and file.split('.')[0] < o2 :
                            image = np.load(path + '/' + file)
                            err = abs(degree-image.shape[1])
                            if err < n and image.shape[1]>3:
                                a = file
                                n = err
                    if n != 100 and m != 0:
                        if d == 'Full':
                            if a.split('.')[0] >= '2021-5-21-12-12-26-43' and a.split('.')[0] <= '2021-5-21-12-12-27-30':
                                cls = 'Action'
                            elif a.split('.')[0] >= '2021-5-21-12-12-27-30':
                                cls = 'Submerge'
                            else:
                                cls = 'Normal'
                        else:
                            cls = d
                        image = np.load(path + '/' + a)
                        print(sonar, cls, o, a)
                        print(image[:deg, :, :].shape)
                        np.save('5/' + sonar + 'images/'+ cls + '_' + str((r[2]+r[1])/2) + '-' + str((r[4]+r[3])/2) + '_' + a.split('.')[0] + '.npy', image[:deg, :, :])
    elif args.exp == 6:
        # use timestamp to classify
        num_sample = 500
        scan_range = 20
        sonar_image_refs = {}
        i = 0
        # the timestamp of Action and Submerge
        Action = [['09-59-10', '10-01-03'], ['10-14-44', '10-16-03'], ['10-33-34', '10-34-35'], ['10-44-53', '10-46-46'], ['11-03-19', '11-04-47'], ['11-10-24', '11-12-39']]
        Submerge = [['10-04-16', '10-07-03'], ['10-17-40', '10-20-18'], ['10-38-40','10-40-08'], ['10-52-20', '10-53-07'], ['11-06-31', '11-07-59'], ['11-14-12', '11-18-13']]
        for sonar in ["sonar_1/", "sonar_2/"]:
            f = open('6/' + sonar + 'update_schedule.txt', 'w')
            shutil.rmtree('6/' + sonar + 'images/')
            os.mkdir('6/' + sonar + 'images/')
            path = '6/' + sonar + 'mode_2'
            files = os.listdir(path)
            for file in files:
                a, b = file.split('.')
                if b == 'npy':
                    r, t = a.split('_')
                    r = r.split('-')
                    degree_span = int(r[1]) - int(r[0])
                    distance_span = int(r[3]) - int(r[2])
                    label, d = time2label1(t[-8:], Action, Submerge)
                    if degree_span > 4 and int(r[3]) < 250:
                        if (d == 1 and int(r[2]) > 150) or (d == 0 and int(r[3]) < 150):
                            crop_image = np.load(path + '/' + file)[int(r[2]):int(r[3]), :, :]
                            np.save('6/' + sonar + 'images/' + str(label) + '_' + a + '.npy', crop_image)
                            plt.scatter(degree_span, int(r[3]))
                            i = i + 1
                else:
                    f.write(a + "\n")
            f.close()
        print(i)
        plt.show()
    elif args.exp == 7:
        # use timestamp to classify
        num_sample = 500
        scan_range = 20
        sonar_image_refs = {}
        i = 0
        # the timestamp of Action and Submerge = START END
        # drown_swim, drown_still, drown_drown, overlap, noise_pattern, fourpeople
        time_stamp = [['10-30-36', '10-30-59'], ['10-33-23', '10-34-21'], ['10-34-31', '10-35-23'], ['10-37-50', '10-39-31'], ['10-44-25', '10-46-00'], ['10-47-50', '10-49-17'],
                      ['10-52-00', '10-52-27'], ['10-53-13', '10-55-24'], ['10-56-48', '10-59-14'],
                      ['11-01-13', '11-03-22'],
                      ['11-10-46', '11-11-40'], ['11-13-10', '11-14-04'], ['11-14-53', '11-16-23'],
                      ['11-20-57', '11-23-46'],
                      ['11-35-13', '11-36-18']]
        f = open('7/' + 'update_schedule.txt', 'w')
        shutil.rmtree('7/' + 'images/')
        os.mkdir('7/' + 'images/')
        for folder in ['mode_2/', 'overlap/', 'fourpeople/']:
        # for folder in ['noise_pattern/']:
            files = os.listdir('7/' + folder)
            for file in files:
                a, b = file.split('.')
                if b == 'npy':
                    r, t = a.split('_')
                    r = r.split('-')
                    t = t[-8:]
                    c = None
                    for d in range(len(time_stamp)):
                        if t < time_stamp[d][1] and t > time_stamp[d][0]:
                            c = d
                    if c:
                        label = time2label2(c, r)
                    else:
                        label = 0
                    degree_span = int(r[1]) - int(r[0])
                    distance_span = int(r[3]) - int(r[2])
                    #if degree_span > 2 and int(r[3]) < 320 and int(r[2]) > 50:
                    if degree_span > 3 and int(r[3]) < 300 and int(r[2]) > 50:
                        crop_image = np.load('7/' + folder + file)[int(r[2]):int(r[3]), :, :]
                        np.save('7/images/' + str(label) + '_' + a + '.npy', crop_image)
                        #plt.scatter(degree_span, int(r[3]), c='blue')
                        i = i + 1
                    else:
                        #plt.scatter(degree_span, int(r[3]), c='red')
                        pass
                else:
                    f.write(a + "\n")
        f.close()
        print(i)
        # plt.show()

    else:# all random data
        n, m1, m2, w = 0, 0, 0, 0
        num_sample = 500
        scan_range = 20
        sonar_image_ref = {}
        directories = ["3/sonar_1/swim", "3/sonar_1/walk", "3/sonar_2/swim", "3/sonar_2/walk", "4/sonar_1/Random", "4/sonar_2/Random"]
        for ref in ["3/sonar_1", "3/sonar_2", "4/sonar_1", "4/sonar_2"]:
            sonar_image_ref[ref] = np.zeros((num_sample, 400))
            reference = ref + "/reference.txt"
            f = open(reference, "r")
            lines = f.readlines()
            for line in lines:
                angle, data = readline(line)
                if len(data) == num_sample:
                    sonar_image_ref[ref][:, angle] = data
            f.close()
        for d in directories:
            s = os.path.dirname(d)
            files = os.listdir(d)
            for file in files:
                if file.split('.')[-1] == 'npy':
                    continue
                f = open(d + '/' + file, 'r')
                lines = f.readlines()
                sonar_img = np.zeros((num_sample, 400))
                angle_former = 0
                object_former = {}
                object_record = {}
                peaks_record = [[[], []]] * 400
                for i in range(len(lines)):
                    if i >= 1:
                        angle_former = angle
                    angle, data = readline(lines[i])
                    if len(data) == 0:
                        continue
                    data = abs(data - sonar_image_ref[s][:, angle])
                    len_sample = scan_range / len(data)
                    data_filter = smooth(data, len_sample, 0)
                    local_var = smooth(abs(data - data_filter), len_sample, 1)
                    peaks, dict = detect(data_filter, len_sample, local_var)
                    new_object, overlap = update_record(peaks_record, object_record, dict, angle, angle_former,len_sample)
                    sonar_img[:, angle] = data_filter
                    # for continuously scan
                    rmax = 0
                    m1_add = 0
                    m2_add = 0
                    for o in object_former:
                        if o[1] not in overlap:
                            if filter(object_record[o], threshold) and (object_record[o][2]-object_record[o][1])>3:
                                m1_add = 1
                                if object_record[o][4] > rmax:
                                    rmax = object_record[o][4]
                                    r = object_record[o]

                        else:
                            m2_add = 1
                    if rmax != 0:
                        plt.scatter(r[2] - r[1], r[4] - r[3])
                    m1 = m1 + m1_add
                    m2 = m2 + m2_add
                    object_former = new_object
                    angle_former = angle
                w = w + len(lines)
                f.close()
        # object done, object ing, nothing
        print(m1/w, m2/w, (w-m1-m2)/w)
        plt.show()







