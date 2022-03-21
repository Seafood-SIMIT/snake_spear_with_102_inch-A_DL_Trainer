import torch
import torch.nn as nn
#from mir_eval.separation import bss_eval_sources

import numpy as np
#正交匹配追踪
#para:输入信号、字典、迭代次数

def validate(model, testloader, writer, step,  hp,device):
    model.eval()
    
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    #print("atom_mask:", mask_atom, "shape:",mask_atom.shape)
    #print("Parament_test_model:")
    #for parameters in model.parameters():
        #print(parameters)
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
            acc_count_test+=torch.sum(torch.argmax(predict)==test_label,axis=0)

            #sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
        writer.log_evaluation(acc_count_test/len(testloader),loss_sum/len(testloader),loss_sum,step)
            #break
        print('test_loss:', loss_sum/len(testloader), 'testacc:',acc_count_test/len(testloader))
            

    model.train()
    return acc_count_test/len(testloader)