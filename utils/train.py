#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:28:48 2020

@author: sunlin
"""

import os
import math
import torch
import torch.nn as  nn
import traceback
from torch.nn import init
import torch.optim as optim
from utils.adabound import AdaBound
from utils.evaluation import validate
import numpy as np

def train(args, pt_dir, chkpt_path, trainloader, testloader, writer,model, logger, hp, valid_model,device):
    #load embedder
    model.to(device)
    
    #选定学习策略
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(),
                             lr=hp.train.adabound.initial,
                             final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=hp.train.adam)
    elif hp.train.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr = hp.train.sgd) 
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)
    
    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['filter_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        
        #使用新超参数
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")
        
    try:
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss()
        all_length_frame = len(trainloader)*hp.train.batch_size
        for step in range(hp.train.epoch):
            model.train()
            loss_sum=0
            train_acc = 0
            #print(filter_model.fc1[0].weight.requires_grad)
            for data_input, target_label in trainloader:
                data_input,target_label = [item.to(device) for item in (data_input,target_label.long())]
                predict = model(data_input)
                #print("InputSize:",mask_input.dtype, "LabelSize:",target_label.dtype)
                loss = criterion(predict, target_label)   #计算loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_sum = loss_sum+ loss.item()
                
                #print(predict,'\n',target_label)
                #loss.retain_grad()
                #target_label_cpu=target_label.cpu()
                #计算train准确率
                #for i in range(hp.train.batch_size):
                #    if torch.argmax(predict[i]) == target_label[i]:
                #        train_acc+=1
                train_acc+=torch.sum(torch.argmax(predict)==target_label,axis=0)
    
            logger.info("[Epoch] %d " % step)

            logger.info('train_loss:%.6f, train_acc:%.4f'%(loss_sum/all_length_frame, train_acc/all_length_frame))
            writer.log_training(loss_sum,loss_sum/all_length_frame, train_acc/all_length_frame, step)
            
            if valid_model:
                continue
            val_acc=validate(model, testloader, writer, step, hp,device)
            if step % hp.train.checkpoint_interval == 0:
                save_path = os.path.join(pt_dir, '%s_checkout_%dstep[%.2f].pt' % (args.model,step, val_acc))
                torch.save({
                    'model':model.state_dict(),
                    #'optimizer':optimizer.state_dict(),
                    #'step':step,
                    #'hp_str':hp_str,
                    },save_path
                    )
            
            
    except Exception as e:
        logger.info("Exiting due to exception: %s" %e)
        traceback.print_exc()
