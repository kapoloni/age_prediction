#!/usr/bin/python3
import os

# mode = 'LRF'
mode = 'Train'
age = '[0,70]'

clr = {'L_RMS': '[-4.45,-4.15]', 'R_RMS': '[-4.45,-4.15]',
       'L_Adam': '[-4.3,-2.2]', 'R_Adam': '[-4.3,-1.6]',
       'L_SGDm': '[-2.7,0]', 'R_SGDm': '[-2.7,-0.3]'}

# for dp in np.arange(0.2, 0.7, 0.1):
for dp in ['0.3', '0.4']:  # , 'Adam', 'SGDm'
    # dp = dp.round(2)
    # print(dp)
    for side in ['R', 'L']:
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
                train = './train.py --num_epochs 400 --batch_size 512' + \
                        ' --loss MAE --side ' + side + ' --age_range ' + \
                        age + ' --optimizer ' + optimizer + \
                        ' --weight_decay 0 --clr ' + \
                        clr[side + "_" + optimizer] + \
                        ' --dropout_rate ' + str(dp)
                print(train)
                os.system(train)
