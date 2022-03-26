import torch
import torch.nn as nn
#from mir_eval.separation import bss_eval_sources

import numpy as np
#正交匹配追踪
#para:输入信号、字典、迭代次数

def validate(logger,model, testloader, writer, step,  hp,device):
    model.eval()
    
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    #print("atom_mask:", mask_atom, "shape:",mask_atom.shape)
    #print("Parament_test_model:")
    #for parameters in model.parameters():
        #print(parameters)
    all_length_frame = len(testloader)*hp.train.batch_size
    with torch.no_grad():
        loss_sum =0
        acc_count_test=0
        for test_energy,test_label in testloader:
            test_energy,test_label = [item.to(device) for item in (test_energy,test_label.long())]
            #print(len(batch_item))
            
            #test_energy = test_energy/torch.norm(test_energy, p = 2, dim = 1, keepdim=True)
            predict = model(test_energy)
            test_loss = criterion(predict, test_label)
            
            loss_sum = loss_sum+ test_loss.item()
            #acc_count_test+=torch.sum(torch.argmax(predict)==test_label,axis=0)
            for i in range(hp.train.batch_size):
                if torch.argmax(predict[i]) == test_label[i]:
                    acc_count_test+=1

            #sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
        #writer.log_evaluation(acc_count_test/all_length_frame ,loss_sum/all_length_frame)
            #break
        logger.info('test_loss:%.6f, test_acc:%.4f'%(loss_sum/all_length_frame, acc_count_test/all_length_frame))
            

    model.train()
    return acc_count_test/all_length_frame
