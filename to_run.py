#!/usr/bin/python3
import sys
import os
import time

# Learning rate finder

# Config

age = '[0,70]'

clr = {'L_SGD': '[-2.0,0.]', 'R_SGD': '[-2.0,0.]',
       'L_SGDm': '[-2.9,-0.3]', 'R_SGDm': '[-2.9,-0.4]',
       'L_Adam': '[-4.3,-2.5]', 'R_Adam': '[-4.3,-2.2]',
       'L_RMS': '[-5.2,-4.2]', 'R_RMS': '[-5.2,-4.2]'}

dp = {'L': '0.4', 'R': '0.5'}

for dp in ['0.2', '0.3', '0.4', '0.5', '0.6']:
        for side in ['L']:
                for optimizer in ['RMS']:
                        # Finding learning rate - run just once and verify clr limits
                        flr = './find_learning_rate.py --batch_size 512 --loss MAE --side ' + side + \
                        ' --age_range ' + age + ' --optimizer ' + optimizer + \
                        " --dropout_rate " + dp
                        os.system(flr)
#         # for wd in ['10e-6', '10e-5', '10e-4']:
#         wd = '0'
#         # Train
#         train = './train.py --num_epochs 400 --batch_size 512 --loss MSE ' + \
#                 ' --side ' + side + ' --age_range ' + age + ' --optimizer ' + \
#                 optimizer + ' --weight_decay ' + wd + ' --clr ' + \
#                 clr[side + "_" + optimizer] + " --dropout_rate " + dp[side]
#         print(train)
#         os.system(train)

# snapshot = {'L': 'outputs/ckpt_16-04-2021_age_[0-70]_RMS_wd_0_L_dp0.4.pth.tar',
#             'R': 'outputs/ckpt_16-04-2021_age_[0-70]_RMS_wd_0_R_dp0.5.pth.tar'}

# for side in ['L', 'R']:
#     for optimizer in ['RMS']:
#         # Finding learning rate - run just once and verify clr limits
# #         flr = './find_learning_rate.py --batch_size 512 --loss MAE --side ' + side + \
# #               ' --age_range ' + age + ' --optimizer ' + optimizer + \
# #               " --dropout_rate " + dp[side]
# #         os.system(flr)
#         # for wd in ['10e-6', '10e-5', '10e-4']:
#         wd = '0'
#         # Train
#         train = './train.py --num_epochs 150 --batch_size 512 --loss MSE ' +\
#                 '--side ' + side + ' --age_range ' + age + ' --optimizer ' + \
#                 optimizer + ' --weight_decay ' + wd +\
#                 ' --clr ' + clr[side+"_"+optimizer] + \
#                 ' --snapshot ' + snapshot[side] + " --dropout_rate " + dp[side]

#         print(train)
#         os.system(train)
