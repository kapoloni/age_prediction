#!/usr/bin/python3
import os

# mode = 'LRF'
mode = 'Train'
age = '[0,70]'


for dp in ['0.0']:  # , 'Adam', 'SGDm'
    wd = '0'
    for side in ['L']:
        if side == 'L':
            clr = {'L_RMS': '[-4.3,-4]'}
        else:
            clr = {'R_RMS': '[-5.2,-3.4]'}
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 128' + \
                      ' --loss MAE --side ' + side + " --data_aug False " + \
                      ' --age_range ' + age + ' --optimizer ' + optimizer + \
                      ' --weight_decay ' + wd + \
                      ' --dropout_rate ' + str(dp) + \
                      ' --num_workers 20'
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 150 --batch_size 128' + \
                        ' --loss MAE --side ' + side + " --data_aug False " + \
                        ' --age_range ' + age + ' --optimizer ' + optimizer + \
                        ' --weight_decay ' + wd + \
                        ' --dropout_rate ' + str(dp) + \
                        ' --clr ' + clr[side + "_" + optimizer] + \
                        ' --num_workers 20'

                print(train)
                os.system(train)


'''
# WD influence
clr = {'L_RMS': '[-5,-2.7]'}
for wd in ['1e-5']:  # , 'Adam', 'SGDm'
    dp = '0.0'
    for side in ['L']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 256' + \
                      ' --loss MAE --side ' + side + " --data_aug True " + \
                      ' --age_range ' + age + ' --optimizer ' + optimizer + \
                      ' --weight_decay ' + wd + \
                      ' --dropout_rate ' + str(dp) + \
                      ' --num_workers 20'
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 200 --batch_size 256' + \
                        ' --loss MAE --side ' + side + " --data_aug True " + \
                        ' --age_range ' + age + ' --optimizer ' + optimizer + \
                        ' --weight_decay ' + wd + \
                        ' --dropout_rate ' + str(dp) + \
                        ' --clr ' + clr[side + "_" + optimizer] + \
                        ' --num_workers 20'
                print(train)
                os.system(train)


# Da influence
lim1 = {'RMS': [-4, -3.57]}
lim2 = {'RMS': [-4.9, -2.6]}
for da in ['True', 'False']:  # , 'Adam', 'SGDm'
    if da == 'True':
        clr = {'L_RMS': '[-5,-2.9]', 'R_RMS': '[-5,-2.65]'}
    else:
        clr = {'L_RMS': '[-4.,-3.57]', 'R_RMS': '[-4.,-3.57]'}
    dp = '0.0'
    for side in ['R']:
        for optimizer in ['RMS']:
            if mode == 'LRF':
                # Finding learning rate - run just once and verify clr limits
                flr = './find_learning_rate.py --batch_size 256' + \
                        ' --loss MAE --side ' + side + " --data_aug " + da + \
                        ' --age_range ' + age + ' --optimizer ' + optimizer + \
                        ' --dropout_rate ' + str(dp) + " --num_workers 20"
                os.system(flr)
            else:
                # Train
                train = './train.py --num_epochs 200 --batch_size 256' + \
                        ' --loss MAE --side ' + side + " --data_aug " + da + \
                        ' --age_range ' + age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --dropout_rate ' + str(dp) + \
                        ' --clr ' + clr[side + "_" + optimizer] + \
                        ' --num_workers 20'
                print(train)
                os.system(train)
'''
