#!/usr/bin/python3
import os

# mode = 'LRF' 
mode = 'Train'
age = '[0,70]'

clr = {'L_RMS': '[-4.37,-4.09]', 'R_RMS': '[-4.37,-4.09]'}

for dp in ['0.2', '0.3', '0.4', '0.5', '0.6']:
    for side in ['R']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 512' + \
                    ' --loss MAE --side ' + side + \
                    ' --age_range ' + age + ' --optimizer ' + optimizer + \
                    ' --dropout_rate ' + dp
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 400 --batch_size 512' + \
                        ' --loss MSE --side ' + side + ' --age_range ' + \
                        age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --clr ' + \
                        clr[side + "_" + optimizer] + " --dropout_rate " + dp
                print(train)
                os.system(train)
