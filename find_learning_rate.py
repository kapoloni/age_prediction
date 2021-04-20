#!/usr/bin/python3
import os
import sys
import torch
import argparse
import torch.nn as nn
from datetime import date
from age_prediction.data_module import MyDataModule
from age_prediction.module_trainer import ModuleTrainer
from age_prediction.callbacks import LearningRateFinder
from age_prediction.models.\
     efficientnet_pytorch_3d import EfficientNet3D as EfNetB0


def parse_args(args):
    """!@brief
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Training brain age')
    parser.add_argument('--database', help='Database folder',
                        default='../datasets', type=str)
    parser.add_argument('--csv_data', help='Csv infos folder',
                        default='../csv_data', type=str)
    parser.add_argument('--side', help='Left or right hippocampus (L or R)',
                        default='L', type=str)
    parser.add_argument('--data_aug', help='Train with dataAugmentation,\
                        default True', default='True')
    parser.add_argument('--age_range', help='Delimit age interval,\
                        use as [min, max]', default=None)
    parser.add_argument('--gpu', help='True or false, (gpu or cpu),\
                        default: gpu', default='True')
    parser.add_argument('--dataParallel', help='Parallelizes the train \
                        batches', default='True')
    parser.add_argument('--snapshot', help='Resume training from a\
                        snapshot (.pth.tar).', default=None)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--loss', help='Loss function (MSE or MAE)',
                        default='MSE', type=str)
    parser.add_argument('--optimizer', help='Optimizer (RMS, Adam, SGDm)',
                        default='Adam', type=str)
    parser.add_argument('--dropout_rate', help='dropout_rate',
                        default='0.2', type=float)

    return parser.parse_args(args)


if __name__ == "__main__":
    # Parse arguments
    args = sys.argv[1:]
    args = parse_args(args)
    print(args)

    # Get hippo side
    if args.side == 'L':
        side = "_L"
    else:
        side = '_R'

    if args.loss == 'MSE':
        loss = nn.MSELoss()
    else:
        loss = nn.L1Loss(reduction='mean')

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

    dataloader = MyDataModule(database=args.database,
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
                                        lr=.01, alpha=0.9,
                                        eps=1e-08, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    momentum=0.9,
                                    lr=0.01)

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

    output_folder = 'outputs/lr_finder'
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    output_prefix = "_" + d1 + "_age_" + '-'.join(str(age_range).split(" ")).\
                    replace(',', '') + "_" + args.optimizer + side + "_dp" + \
                    str(args.dropout_rate)

    print(output_prefix)

    lrf = LearningRateFinder(ModuleTrainer(model))
    lrf.find(data=dataloader, start_LR=1e-10, end_LR=1e+1,
             batch_size=args.batch_size, loss=loss,
             epochs=50,
             optimizer=optimizer, cuda_device=True)
    lrf.save_csv(os.path.join(output_folder,
                              'lr_finder' + output_prefix + '.csv'))
