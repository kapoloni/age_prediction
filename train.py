#!/usr/bin/python3
import argparse
import sys
import os
import torch
import torch.nn as nn
from datetime import date

from age_prediction.models.\
    efficientnet_pytorch_3d import EfficientNet3D as EfNetB0
from age_prediction.dataloader import MyDataLoader
from age_prediction.trainer import ModuleTrainer
from age_prediction.callbacks import ModelCheckpoint, CyclicLR, \
                                      TensorBoardCB, CSVLogger
from age_prediction.metrics import MSE, MAE


def parse_args(args):
    """!@brief
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Training brain age')
    parser.add_argument('--database', help='Database folder',
                        default='../datasets', type=str)
    parser.add_argument('--csv_data', help='Csv infos folder',
                        default='database_split', type=str)
    parser.add_argument('--side', help='Left or right hippocampus (L or R)',
                        default='L', type=str)
    parser.add_argument('--data_aug', help='Train with dataAugmentation,\
                        default True', default='True')
    parser.add_argument('--age_range', help='Delimit age interval, use as [min, max],\
                        default None', default=None)
    parser.add_argument('--gpu', help='True or false, (gpu or cpu),\
                        default: gpu', default='True')
    parser.add_argument('--dataParallel', help='Parallelizes the train \
                        batches', default='True')
    parser.add_argument('--snapshot', help='Resume training from a snapshot \
                        (.pth.tar).', default=None)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--num_epochs', help='Number of epochs', default=250)
    parser.add_argument('--loss', help='Loss function (KLDiv or MAE)',
                        default='KLDiv', type=str)
    parser.add_argument('--optimizer', help='Optimizer (RMS, Adam, SGR)',
                        default='Adam', type=str)
    parser.add_argument('--cyclicalLR', help='cyclicalLR', default=True)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default='1e-5')
    parser.add_argument('--clr', help='CLR limits',
                        default='[-3.1, -1.25]')
    parser.add_argument('--dropout_rate', help='dropout_rate',
                        default='0.2', type=float)

    return parser.parse_args(args)


if __name__ == "__main__":

    # Parse arguments
    args = sys.argv[1:]
    args = parse_args(args)
    print(args)

    # Load images
    if args.side == 'L':
        side = "_L"
    else:
        side = '_R'

    if args.loss == 'MSE':
        loss = nn.MSELoss()
        metrics = [MAE()]
    else:
        loss = nn.L1Loss(reduction='mean')
        metrics = [MSE()]

    if args.age_range is not None:
        age_range = [int(args.age_range.split(",")[0].split("[")[-1]),
                     int(args.age_range.split(",")[-1].split("]")[0])]
        if (age_range[0] == 0) & (age_range[1] == 70):
            train_file = 'train_0-70.csv'
            val_file = 'val_0-70.csv'
        elif (age_range[0] == 70) & (age_range[1] == 100):
            train_file = 'train_70-100.csv'
            val_file = 'val_exp.csv'
        else:
            train_file = 'train_all.csv'
            val_file = 'val_exp.csv'
            print("Using a unknown age range")

    dataloader = MyDataLoader(database=args.database,
                              csv_data=args.csv_data,
                              side=side,
                              batch=args.batch_size,
                              data_aug=eval(args.data_aug),
                              age_range=age_range,
                              train_file=train_file,
                              val_file=val_file
                              )

    dataloader.prepare_data('fit')
    dataloader.setup('fit')

    train_size = len(dataloader.train.inputs[0])

    print("Training with", train_size,
          "evaluating with", len(dataloader.val.inputs[0]))

    # Load effNet3D B0
    model = EfNetB0.from_name("efficientnet-b0",
                              override_params={
                                  'num_classes': 1,
                                  'dropout_rate': args.dropout_rate
                              },
                              in_channels=1,
                              )
    if eval(args.gpu):
        print("Using GPU")
        device = torch.device('cuda')
        if eval(args.dataParallel):
            if torch.cuda.device_count() > 1:
                print(torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)
    else:
        print("Using cpu")
        device = torch.device('cpu')

    model = model.to(device)

    if args.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=.256, alpha=0.9,
                                        eps=1e-08, momentum=0.9,
                                        weight_decay=float(args.weight_decay))
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=.01,
                                     weight_decay=float(args.weight_decay))
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.01, momentum=0.9,
                                    weight_decay=float(args.weight_decay))

    if args.snapshot is not None:
        print('Loading model from {}'.format(args.snapshot))
        checkpoint = torch.load(args.snapshot)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        _loss = checkpoint['loss']
        _val_loss = checkpoint['val_loss']
        print("Snapshot trained for {} epochs. \
               Loss: {} and Val loss {}".format(epoch, _loss, _val_loss))

    output_folder = 'outputs'
    today = date.today()
    # dd/mm/YY
    d1 = today.strftime("%d-%m-%Y")
    output_prefix = d1 + "_age_" + \
        "-".join(str(age_range).split(" ")).replace(',', '') + \
        "_" + args.optimizer + "_wd_" + str(args.weight_decay) + \
        side + "_dp" + str(args.dropout_rate)

    print(output_prefix)
    callbacks = [CSVLogger(file=os.path.join(output_folder,
                           'logger_' + output_prefix + '.csv'))]

    if args.cyclicalLR:
        clr = [float(args.clr.split(",")[0].split("[")[-1]),
               float(args.clr.split(",")[-1].split("]")[0])]
        print('clr limits', clr)
        # 10-3.1, 10-1.25
        step_size = 6*(train_size//args.batch_size)
        print(step_size)
        callbacks.append(CyclicLR(base_lr=10**clr[0],
                                  max_lr=10**clr[1],
                                  mode='triangular2',
                                  step_size=step_size
                                  )
                         )

    callbacks.append(ModelCheckpoint(directory=output_folder,
                                     filename='ckpt_' +
                                              output_prefix))

    callbacks.append(TensorBoardCB(log_dir='_'.join(
                                               output_prefix.split("_")[1:]
                                               ),
                                   max_img_grid=16,
                                   imgs_batch=1))

    #  Train
    trainer = ModuleTrainer(model.to(device))

    trainer.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=metrics,
                    callbacks=callbacks)

    trainer.fit_loader(dataloader.train_dataloader(),
                       dataloader.val_dataloader(),
                       num_epoch=int(args.num_epochs),
                       cuda_device=eval(args.gpu))
