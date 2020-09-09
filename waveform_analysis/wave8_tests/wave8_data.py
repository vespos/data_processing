#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:11:06 2020

@author: mkwiatko
"""
import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PyQt5
# %matplotlib qt

#%%

def get_intensities(f):
    events = file2frames(f)
    wave8data = []
    for event in events:
        dat = descramble_frame(event)
        if dat['Int'].size==0:
            continue
        wave8data.append(dat['Int'])
    return np.asarray(wave8data)


def get_waveforms(f, channel):
    events = file2frames(f)
    wave8data = []
    for event in events:
        dat = descramble_frame(event)
#         if dat['Wave{}'.format(channel)].size==0:
#             continue
        wave8data.append(dat['Wave{}'.format(channel)])
    return np.asarray(wave8data)


def analyze_wave8_wfs(file, channel, n_components):
    wfs = wave8.get_waveforms(file, channel)
    wfs = removeBackground(wfs, ave_window=[0,30])
    regressor = proc.construct_waveformRegressor(wfs[:100], n_components=n_components)
    
    fig = plt.figure(figsize=(15,10))
    fig.suptitle(file.name)
    for ii in range(9):
        idx = np.random.randint(wfs.shape[0])
        ax = fig.add_subplot(3,3,ii+1)
        ax.set_title(str(idx))
        ax.plot(wfs[idx], label='data')
        ax.plot(regressor.fit_reconstruct(wfs[idx]).T, label='fit')
        ax.legend()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    intensity = regressor.get_pulse_intensity(wfs)
    average = np.mean(intensity)
    sigma = np.std(intensity)
    rms = sigma/average
    fig, ax = plt.subplots(ncols=2, figsize=(10,6))
    ax[0].plot(intensity, '.', color='orange')
    ax[0].set_ylabel('Intensity')
    ax[1].set_title('Intensity: norm(components)')
    ax[1].hist(intensity[:,0], bins=30, label='rms={:.05f}'.format(rms))
    ax[1].legend()
    plt.show()

    print('Standard deviation/mean:',rms)
    print('/n/n')
    
    return sigma/average, intensity, wfs, regressor


def file2frames(f):
    '''
    reads rogue binary file and returns a list of events
    '''
    ret = []
    fileOffset = 0
    numberOfFrames = 0
    while True:
        try:
            
            file_header = np.fromfile(f, dtype='uint32', count=2, offset=fileOffset)
            payloadSize = int(file_header[0]/4)-1
            payload = np.fromfile(f, dtype='uint32', count=payloadSize, offset=fileOffset+8)
            fileOffset = fileOffset + file_header[0]+4
            
            ret.append(payload)
            numberOfFrames = numberOfFrames + 1 
            
        except Exception: 
            #e = sys.exc_info()[0]
            #print ("Message\n", e)
            print("Events read: %d"%numberOfFrames)
            break
    
    
    return ret
    
    

#%%

def descramble_frame(data):
    '''
    descramble single event from the back upwards
    packet channel enumeration:
        2  - Raw Waveform 0 (uint16)
        ...
        9  - Raw Waveform 7 (uint16)
        10 - Integrals (uint32)
        11 - Position, Intenfity (float64)
    '''
    ret = {}
    #descramble the frame from the back upwards
    i = len(data)
    while i > 0:
        #find the footer ang get the channel number and size
        size_bytes =  data[i-2]
        channel = data[i-1] & 0xf
        
        if channel == 11:
            ret['ProcQuadSel'] = data[i-10] & 0x1
            tmp = np.copy(data[i-8:i-2])
            tmp.dtype = np.float64
            ret['Proc'] = tmp
        if channel == 10:
            ret['IntBlSize'] = 2**((data[i-11] & 0x70000)>>16)
            ret['IntSize'] = (data[i-12] & 0xff)
            ret['IntDelay'] = (data[i-12] & 0xff00) >> 8
            tmp = np.copy(data[i-10:i-2])
            ret['Int'] = tmp
        if channel <= 9 and channel > 1:
            tmp = np.copy(data[i-int(size_bytes/4)-2:i-2])
            tmp.dtype = np.uint16
            ret['Wave%d'%(channel-2)] = tmp
        if channel < 2:
            break
        
        #print('Channel %d, size %d bytes' %(channel, size_bytes))
        i = i - int(size_bytes/4) - 2
        
    return ret


#%%

# # Read the binary file and find consecutive events

# f = "/u1/mkwiatko/wave8_data/data3.dat"

# events = file2frames(f)


# #%%
# wave8data = descramble_frame(events[0])
# i=0
# channel = 'Wave2'
# stop = len(events)
# fig, ax = plt.subplots(figsize=(12,8),dpi=100)

# x = np.arange(0, len(wave8data[channel]), 1)
# line, = ax.plot(x, wave8data[channel])



# def updatefig(*args):
#     global i
#     if (i<stop):
#         i += 1
#     else:
#         i = 0
#     wave8data = descramble_frame(events[i])
#     line.set_ydata(wave8data[channel])  # update the data.
#     return line,


# ani = animation.FuncAnimation(fig=fig, func=updatefig, interval =1)


# plt.title(channel)
# plt.show()



