#!/usr/bin/python3
import os
import numpy as np

mode = 'LRF'
# mode = 'Train'
age = '[0,70]'

clr = {'L_RMS': '[-4.3,-4.1]', 'R_RMS': '[-4.3,-4.1]'}

for dp in np.arange(0.2, 0.5, 0.1):
    dp = dp.round(2)
    print(dp)
    for side in ['L', 'R']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 512' + \
                      ' --loss MAE --side ' + side + \
                      ' --age_range ' + age + ' --optimizer ' + optimizer + \
                      ' --dropout_rate ' + str(dp)
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 600 --batch_size 512' + \
                        ' --loss MAE --side ' + side + ' --age_range ' + \
                        age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --clr ' + \
                        clr[side + "_" + optimizer] + \
                        ' --dropout_rate ' + str(dp)
                print(train)
                os.system(train)
