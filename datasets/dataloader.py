#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:34:23 2020

@author: sunlin
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import librosa
import pywt

def wpdPlt(signal,wavelet_name='db3',wavelet_depth=8):
    wp = pywt.WaveletPacket(data=signal, 
                            wavelet=wavelet_name,
                            mode='symmetric',
                            maxlevel=wavelet_depth)
    re = []
    for i in [node.path for node in wp.get_level(wavelet_depth, 'freq')]:
        re.append(wp[i].data)
    #能量特征
    energy=[]
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None), 2))
    energy = np.array(energy[0:64])
    energy = energy/np.sum(energy)
    #energy = energy/energy.sum
    #energy = energy/np.sqrt(np.dot(energy,energy.T))
    return energy    
def standLization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
def chooseFeature(data,feature):
    if feature == 'mfcc':
        data = yuchuliAco(data,32)
    elif feature == 'wavelet':
        data = wpdPlt(data,'db3',8)
    elif feature =='none':
        data = data
    else:
        data=data
    return data
def getLabel(file):
    label=-1
    if 'smallwheel' in file:
        label=0
    elif 'largewheel' in file:
        label=1
    elif 'track' in file:
        label=2
    else:
        label=3
    return label
def yuchuliAco(data_aco,input_size=32):
    data_aco=data_aco-np.mean(data_aco)
    mfccs = librosa.feature.mfcc(y=data_aco, sr = 22050, S=None, norm = 'ortho', n_mfcc=input_size)
    return np.mean(mfccs.T,axis = 0)
def readData(hp,valid_model,logger):
    filelist = os.listdir(hp.data.data_dir)
    #print(filelist[0])
    #filelist = list(compress(dir_hp,filelist))                # 找到该目录下所有.mat文件
    assert len(filelist) !=0, \
            "No training file found"
    single_data = []
    single_label = []
    if valid_model:
        index_arrange =random.sample(filelist,4)
    else:
        index_arrange =filelist
    for file in tqdm(index_arrange):
        if file.startswith('.'):
            continue
        #print(index,'/',len(filelist),file)
        label=getLabel(file)
        file = os.path.join(hp.data.data_dir,file)
        #print(file)
        origin_signal = np.loadtxt(file)
        if hp.method.data_type == 'aco':
            origin_signal = origin_signal[:,0].reshape(-1)
        elif hp.method.data_type != 'seis':
            logger.error("Wrong Param of data_type")
        origin_signal = origin_signal[::8]
        for i in range(len(origin_signal)//hp.data.frame_length-1):
            frame_data = standLization(origin_signal[i*hp.data.frame_length:hp.data.frame_length*(i+1)])
            frame_data = chooseFeature(frame_data,hp.method.feature)
            single_data.append(frame_data)
            single_label.append(label)
        #if valid_model:
        #    return np.array(single_data)[0:3,:],np.array(single_label)[0:3]
    return np.array(single_data),np.array(single_label)
#制作dataloader
def createDataloader(hp,valid_model,logger):
    dataset,label = readData(hp,valid_model,logger)
    #label = np.argmax(label,axis=1)
    test_size = 0.1 if valid_model else hp.data.test_size
    train_set, test_set,train_label,test_label = train_test_split(dataset, label, test_size = test_size,random_state = 0)
    logger.info("trainSetShape:%s, trainLabelShape:%s, testSetShape:%s, testLabelShape:%s"%(train_set.shape, train_label.shape,test_set.shape,test_label.shape))
    trainloader = DataLoader(dataset=SFDataset(train_set,train_label),
                        batch_size = hp.train.batch_size,
                        #batch_size=1,
                        shuffle=True,
                        #num_workers=hp.train.num_workers,
                        #collate_fn= train_collate_fn,
                        #pin_memory=True,
                        drop_last = True,
                        #sampler=None
                        )
    validloader = DataLoader(dataset=SFDataset(test_set,test_label),
                        #collate_fn=test_collate_fn,
                        batch_size=hp.train.batch_size, shuffle=False, num_workers=0,
                        drop_last = True)
    return trainloader, validloader
    
    
class SFDataset(Dataset):
    def __init__(self, data, label):
        self.length = len(label)
        self.tensor_data = torch.tensor(data, dtype=torch.float)
        self.tensor_label = torch.tensor(label, dtype=torch.float)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.tensor_data[idx], self.tensor_label[idx]
        #return 0, 0
            
        
    
    
