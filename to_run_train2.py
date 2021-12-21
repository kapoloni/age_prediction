#!/usr/bin/python3
import os

# Learning rate finder
# mode = 'LRF'

mode = 'Train'
age = '[70,100]'

snapshot = {'L': 'outputs/results/ckpt_10-06-2021_age_[0-70]_RMS_wd_' +
                 '0_L_dp0.2_model_best_clr_[-5.2,-3.7].pth.tar',
            'R': 'outputs/results/ckpt_10-06-2021_age_[0-70]_RMS_wd_' +
                 '0_R_dp0.3_model_best_clr_[-5.2,-3.5].pth.tar'}

# clr = {'L_RMS': '[-5.2,-3.7]', 'R_RMS': '[-4.9,-3.7]'}
clr = {'L_RMS': '[-4.7,-3.3]', 'R_RMS': '[-4.7,-3.3]'}
# clr = {'L_RMS': '[-5,-2.8]', 'R_RMS': '[-4.8,-2.9]'}

for dp in ['0.2']:
    for side in ['L']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 128' + \
                      ' --loss MAE --weight_decay 0 --side ' + side + \
                      ' --age_range ' + age + ' --optimizer ' + optimizer + \
                      ' --dropout_rate ' + dp + \
                      ' --snapshot ' + snapshot[side] + \
                      ' --num_workers 20'
                print(flr)
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 100 --batch_size 128' + \
                        ' --loss MAE --side ' + side + ' --age_range ' + \
                        age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --clr ' + \
                        clr[side + "_" + optimizer] + \
                        " --dropout_rate " + dp + \
                        ' --snapshot ' + snapshot[side] + \
                        ' --num_workers 20'
                print(train)
                os.system(train)
