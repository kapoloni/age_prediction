#!/usr/bin/python3
import sys
import os
import time

# Learning rate finder

age = '[70,100]'

snapshot = {'L': 'outputs/ckpt_16-04-2021_age_[0-70]_RMS_wd_0_L_dp0.4.pth.tar',
            'R': 'outputs/ckpt_16-04-2021_age_[0-70]_RMS_wd_0_R_dp0.5.pth.tar'}

clr = {'L_RMS': '[-7.,-3.5]', 'R_RMS': '[-7.,-3.5]'}
# dp = {'L': '0.4', 'R': '0.3'}

for dp in ['0.1', '0.2', '0.3', '0.4', '0.5']:
    for side in ['L', 'R']:
        for optimizer in ['RMS']:
#             Finding learning rate - run just once and verify clr limits
#             flr = './find_learning_rate.py --batch_size 512 --loss MAE --side ' + side + ' --age_range ' + age +\
#                    ' --optimizer ' + optimizer + ' --snapshot ' + snapshot[side] 
#             print(flr)
#             os.system(flr)
#                 for wd in ['10e-6', '10e-5', '10e-4']:
            wd = '0'
#             # Train
            train = './train.py --num_epochs 150 --batch_size 512 --loss MSE ' +\
                         '--side ' + side + ' --age_range ' + age + ' --optimizer ' + \
                         optimizer + ' --weight_decay ' + wd +\
                         ' --clr ' + clr[side+"_"+optimizer] + \
                         ' --snapshot ' + snapshot[side] + " --dropout_rate " + dp
            print(train)
            os.system(train)
# break
    
