# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:21:55 2020

@author: HeLix
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
# import cv2
import os


def get_data(decoded_message):
    data = [int(item) for item in decoded_message.data]
    angle = decoded_message.angle
    return data, angle

def read_txt(path):
    sonar_data = np.zeros((400, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            if len(data) == 500:
                sonar_data[int(angle)] = data
    return sonar_data, int(start_angle), int(end_angle)

def readline(line):
    line = line.split()
    line = list(map(float, line))
    angle = line[0]
    data = line[1:]
    return angle, data

def show_sonar(data2D, distance):
    num_samples, num_degree = np.shape(data2D)
    r = np.linspace(0, distance, num_samples)
    theta = np.linspace(0, 2 * np.pi, num_degree)
    theta, r = np.meshgrid(theta, r)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    plt.pcolormesh(X, Y, data2D, shading='auto', cmap='Greys', antialiased=True)
    K = 40
    for i in range(K):
        angle = np.pi / K * 2 * i
        plt.plot([0, distance * np.cos(angle)], [0, distance * np.sin(angle)], linewidth=0.5, color='red',
                 linestyle="--")

def draw_one_pic(data, save_path):
    w = 1000
    h = 800
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(data, cmap="Greys")
    plt.savefig(save_path, bbox_inches=0)
    # plt.show()
    plt.close()


def temporal_info(data2D, freq, vertical_s=10, bins_freq=3, num_figure=1, mode=0):
    # argument
    # data2D： (num_samples, num_time) raw data from sonar
    # freq: frequency vector
    # vertical_span, bins_freq: only for display
    # num_figure: the object we need to show, include some strong noise
    # mode: show frequency or shape
    d1, d2 = np.shape(data2D)
    # data2D = compensate(data2D)
    fft_data = abs(np.fft.fft(data2D, axis=1))
    sum_data = np.sum(data2D, axis=1)
    for i in range(num_figure):
        highlight = np.argmax(sum_data)
        vertical_span = vertical_s
        if (highlight - vertical_span) < 0:
            vertical_span = highlight
        elif (highlight + vertical_span) >= d1:
            vertical_span = d1 - highlight - 1
        sum_data[highlight - vertical_span:highlight + vertical_span] = 0
        if mode == 0:
            crop_fft_data = np.fft.fftshift(fft_data[highlight - vertical_span:highlight + vertical_span, :], axes=1)
            plt.subplot(num_figure, 2, 2 * i + 1)
            plt.imshow(data2D[highlight - vertical_span:highlight + vertical_span, :], cmap='gray',
                       aspect=d2 / (2 * vertical_span))
            plt.yticks([2 * vertical_span - 1, vertical_span, 0],
                       [highlight + vertical_span - 1, highlight, highlight - vertical_span])
            plt.xlabel('beams')
            plt.ylabel('sample')
            plt.subplot(num_figure, 2, 2 * i + 2)
            plt.imshow(crop_fft_data, cmap='gray', aspect=d2 / (2 * vertical_span))
            plt.yticks([2 * vertical_span - 1, vertical_span, 0],
                       [highlight + vertical_span - 1, highlight, highlight - vertical_span])
            plt.xticks(np.arange(d2)[::int(d2 / bins_freq)], freq[::int(d2 / bins_freq)])
            plt.xlabel('freq')
            plt.ylabel('sample')
            plt.suptitle('scan result and its frequency component')
        else:
            plt.subplot(num_figure, 1, i + 1)
            for j in range(d2):
                plt.plot(range(highlight - vertical_span, highlight + vertical_span),
                         data2D[highlight - vertical_span:highlight + vertical_span, j])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--sonar_num', default='1', type=str)
    parser.add_argument('--batch_id', default='1110004', type=str)
    parser.add_argument('--mode', type=int, default=3,
                        help="0-time control scan result, 1-count control scan result, 2-tracking schemes")

    parser.add_argument('--polar_pic_path', required=False, type=str, default="polar_pic",
                        help="sonar polar picture save path ")
    parser.add_argument('--xy_pic_path', required=False, type=str, default="xy_pic",
                        help="sonar xy picture save path")

    args = parser.parse_args()
    batch_id = args.batch_id
    sonar_num = args.sonar_num

    sonar_data_path = os.path.join('mode_' + str(args.mode), batch_id, 'sonar' + sonar_num)
    polar_pic_path = os.path.join(args.polar_pic_path, batch_id, 'sonar' + sonar_num)
    xy_pic_path = os.path.join(args.xy_pic_path, batch_id, 'sonar' + sonar_num)

    files = os.listdir(sonar_data_path)
    for file in files:
        if file[0] == '.':
            continue
        else:
            sonar_data_file = os.path.join(sonar_data_path, file)
            f = open(sonar_data_file, 'r')
            temp_data, start_angle, end_angle = read_txt(sonar_data_file)

            ## draw polar pictures
            sonar_tmp_data = np.transpose(temp_data)
            show_sonar(sonar_tmp_data, 20)
            os.makedirs(polar_pic_path, exist_ok=True)
            plt.savefig(os.path.join(polar_pic_path, str(file[:-4]) + '.png'))
            # plt.show()
            plt.clf()

            ## draw xy sonar pictures
            os.makedirs(xy_pic_path, exist_ok=True)
            draw_one_pic(temp_data, os.path.join(xy_pic_path, str(file[:-4]) + '.png'))
            # plt.show()
            plt.clf()

