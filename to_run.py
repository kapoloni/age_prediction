#!/usr/bin/python3
import os
import numpy as np

# mode = 'LRF'
mode = 'Train'
age = '[0,70]'

clr = {'L_RMS': '[-5.,-3.]', 'R_RMS': '[-5.,-3.]',
       'L_Adam': '[-4.3,-2.2]', 'R_Adam': '[-4.3,-1.6]',
       'L_SGDm': '[-2.7,0]', 'R_SGDm': '[-2.7,-0.3]'}

for dp in np.arange(0.2, 0.7, 0.1):
# for dp in ['0.2']:  # , 'Adam', 'SGDm'
    dp = dp.round(2)
    # print(dp)
    for side in ['L', 'R']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 256' + \
                        ' --loss MAE --side ' + side + \
                        ' --age_range ' + age + ' --optimizer ' + optimizer + \
                        ' --dropout_rate ' + str(dp) + " --num_workers 20"
                os.system(flr)
            else:
                if dp in [0.5, 0.6]:
                    clr.update({'L_RMS': '[-5.,-3.2]',
                                'R_RMS': '[-5.,-3.8]'})
                # Train
                train = './train.py --num_epochs 400 --batch_size 256' + \
                        ' --loss MAE --side ' + side + ' --age_range ' + \
                        age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --clr ' + \
                        clr[side + "_" + optimizer] + \
                        ' --dropout_rate ' + str(dp) + " --num_workers 20"
                print(train)
                os.system(train)
