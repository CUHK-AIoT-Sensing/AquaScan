# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:21:55 2020

@author: HeLix
"""
import numpy as np
import matplotlib.pyplot as plt
from data import readline
def get_data(decoded_message):
    data=[int(item) for item in decoded_message.data]
    angle=decoded_message.angle
    return data, angle
def show_sonar(data2D,distance):
    num_samples, num_degree=np.shape(data2D)
    r=np.linspace(0,distance,num_samples)
    theta=np.linspace(0,2*np.pi,num_degree)
    theta,r=np.meshgrid(theta,r)
    X=r*np.cos(theta)
    Y=r*np.sin(theta)
    plt.pcolormesh(X,Y,data2D, shading= 'auto', cmap='Greys',antialiased=True)
    K = 40
    for i in range(K):
        angle = np.pi/ K * 2 * i
        plt.plot([0, distance * np.cos(angle)], [0, distance * np.sin(angle)], linewidth = 0.5, color = 'red', linestyle = "--")

def temporal_info(data2D, freq, vertical_s=10, bins_freq=3, num_figure=1, mode=0):
    # argument
    # data2D： (num_samples, num_time) raw data from sonar
    # freq: frequency vector
    # vertical_span, bins_freq: only for display
    # num_figure: the object we need to show, include some strong noise
    # mode: show frequency or shape
    d1, d2= np.shape(data2D)
    #data2D = compensate(data2D)
    fft_data=abs(np.fft.fft(data2D,axis=1))
    sum_data=np.sum(data2D,axis=1)
    for i in range(num_figure):
        highlight=np.argmax(sum_data)
        vertical_span = vertical_s
        if (highlight-vertical_span)<0:
            vertical_span = highlight
        elif (highlight+vertical_span)>=d1:
            vertical_span = d1 - highlight -1
        sum_data[highlight-vertical_span:highlight+vertical_span]=0
        if mode==0:
            crop_fft_data=np.fft.fftshift(fft_data[highlight-vertical_span:highlight+vertical_span,:],axes=1)
            plt.subplot(num_figure,2,2*i+1)
            plt.imshow(data2D[highlight-vertical_span:highlight+vertical_span,:],cmap='gray',aspect=d2/(2*vertical_span))
            plt.yticks([2*vertical_span-1,vertical_span,0], [highlight+vertical_span-1,highlight,highlight-vertical_span])
            plt.xlabel('beams')
            plt.ylabel('sample')
            plt.subplot(num_figure,2,2*i+2)
            plt.imshow(crop_fft_data,cmap='gray',aspect=d2/(2*vertical_span))
            plt.yticks([2*vertical_span-1,vertical_span,0], [highlight+vertical_span-1,highlight,highlight-vertical_span])
            plt.xticks(np.arange(d2)[::int(d2/bins_freq)], freq[::int(d2/bins_freq)])
            plt.xlabel('freq')
            plt.ylabel('sample')
            plt.suptitle('scan result and its frequency component')
        else:
            plt.subplot(num_figure,1,i+1)
            for j in range(d2):
                plt.plot(range(highlight-vertical_span,highlight+vertical_span),data2D[highlight-vertical_span:highlight+vertical_span,j])


if __name__=='__main__':
    path = '6/sonar_2/mode_0/2021-07-29-09-51-59.txt'
    f = open(path, 'r')
    lines = f.readlines()
    sonar_img = np.zeros((500, 400))
    for i in range(len(lines)):
        angle, data = readline(lines[i])
        if len(data) != 0:
            sonar_img[:, i] = data
    show_sonar(sonar_img, 20)
    plt.show()
