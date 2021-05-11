#!/usr/bin/python3
import os

# Learning rate finder
# mode = 'LRF'

mode = 'Train'
age = '[70,100]'

snapshot = {'L': 'outputs/ckpt_26-04-2021_age_[0-70]_RMS_' +
                 'wd_0_L_dp0.6_model_best.pth.tar',
            'R': 'outputs/ckpt_30-04-2021_age_[0-70]_RMS_' +
                 'wd_0_R_dp0.2_model_best.pth.tar'}

clr = {'L_RMS': '[-6,-3]', 'R_RMS': '[-6,-3]'}

for dp in ['0.2', '0.3', '0.4', '0.5', '0.6']:
    for side in ['L', 'R']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 512' + \
                      ' --loss MSE --side ' + side + \
                      ' --age_range ' + age + ' --optimizer ' + optimizer + \
                      ' --dropout_rate ' + dp + ' --snapshot ' + snapshot[side]
                print(flr)
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 200 --batch_size 512' + \
                        ' --loss MSE --side ' + side + ' --age_range ' + \
                        age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --clr ' + \
                        clr[side + "_" + optimizer] + \
                        " --dropout_rate " + dp + \
                        ' --snapshot ' + snapshot[side]
                print(train)
                os.system(train)
