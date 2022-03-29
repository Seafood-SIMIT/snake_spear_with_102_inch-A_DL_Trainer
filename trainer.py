#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:13:10 2020

@author: sunlin
"""

import os
import time
import torch
import random
import logging
import argparse

from utils.train import train
from utils.hparams import HParam,logAndCheckpoinrDirConfig
from utils.writer import MyWriter
from datasets import *
from model import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,help="yaml file for configuration")
    parser.add_argument('-v', '--valid_model', type=bool, required=None,help="Valid the model is functional")
    parser.add_argument('--checkpoint_path', type=str, default=None,help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                       help="Name of the model. Used for both logging and saving checkpoints.")
    args = parser.parse_args()
    
    #读入超参数
    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        #存储超参数为string
        hp_str = ''.join(f.readlines())
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, hp.log.log_dir, args.model)
    pt_dir = os.path.join(base_dir, hp.log.chkpt_dir, args.model)
    logAndCheckpoinrDirConfig(base_dir,hp.log.chkpt_dir, args.model, log_dir,pt_dir)
    
    chkpt_pth = args.checkpoint_path if args.checkpoint_path is not None else None
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
            ]
        )
    logger = logging.getLogger()

    #验证模型阶段 
    valid_model = args.valid_model
    #输入路径检查
    logger.info("Random data file display: %s "%(random.choice(os.listdir(hp.data.data_dir))))
        
    writer = MyWriter(hp, log_dir)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = hp.valid.batch_size if valid_model else hp.train.batch_size
    logger.info("Device Choice: %s"%(device_name))
    device = torch.device(device_name)
    logger.info("Start Reading Data")
    trainloader,testloader = createDataloader(valid_model,logger,batch_size,hp.data.valid_size)
    logger.info("Down read DataSet")
    
    #model = wavaletAcoModel()
    #model = ACOClassifierLSTM(hp.lstm.input_size,hp.lstm.hidden_size,hp.lstm.num_layers)
    #exec("model = "+hp.method.model+"("+hp.method.model_params+")")
    
    model=eval("%s(%s)"%(hp.method.model_name,hp.method.model_param))
    logger.info("Model loaded")
    print(model)
    train(args, pt_dir, chkpt_pth, trainloader, testloader, writer, model,
          logger, hp, valid_model,device)
