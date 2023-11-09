from dataset import *
from loss import *
from lars import *
from model import *
from utils import *
from main import linear_eval, init
import main

import argparse
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule



def run(args):
    ######### INIT #########
    # INIT DEVICES
    if args.hardware == 'multi-gpu':
        if args.computer == 'jeanzay':
            global_rank, local_rank, world_size, is_master, device = init()
        else:
            raise Exception('Multi-GPU is only supported on JeanZay')
    else:
        global_rank, local_rank, world_size, is_master = 0, 0, 1, True
        device = torch.device('cuda') if args.hardware == 'mono-gpu' else torch.device('cpu')
    
    if args.hardware != 'cpu': torch.backends.cudnn.benchmark = True

    # INIT CHECKPOINT
    if args.checkpoint is None:
        exit(-1)  # Can't eval without checkpoint...
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # INIT TRACKER
    main.tracker = Tracker(device)

    # INIT WRITER
    if is_master:
        log_dir = checkpoint['log_dir'] + '_eval'
        main.writer = SummaryWriter(log_dir)

        print('\nArgs:', args, '\n')

    
    ######### DATASETS #########
    # CREATE DATASETS
    train_clsf_set = get_dataset(args.dataset_name, 'train_clsf')
    eval_clsf_set = get_dataset(args.dataset_name, 'eval_clsf')

    # CREATE SAMPLERS
    if args.hardware == 'multi-gpu':
        train_clsf_sampler = torch.utils.data.distributed.DistributedSampler(
            train_clsf_set,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
        eval_clsf_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_clsf_set,
            num_replicas=world_size,
            rank=global_rank
        )
    else:
        train_clsf_sampler = torch.utils.data.RandomSampler(
            train_clsf_set
        )
        eval_clsf_sampler = torch.utils.data.SequentialSampler(
            eval_clsf_set
        )
    
    # CREATE DATALOADERS
    train_clsf_loader = torch.utils.data.DataLoader(
        train_clsf_set,
        batch_size=args.batch_size_clsf//world_size,
        drop_last=True,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=train_clsf_sampler
    )
    eval_clsf_loader = torch.utils.data.DataLoader(
        eval_clsf_set,
        batch_size=args.batch_size_clsf//world_size,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=eval_clsf_sampler
    )


    ######### MODEL and CO #########
    # CREATE MODEL
    model = BYOL(args.resnet_type, args.z_dim, args.dataset_name=='cifar10').to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    if args.hardware == 'multi-gpu':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # CREATE SCALER
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'mixed' else None


    ######### ITERATE EPOCHS #########
    linear_eval(args, model, train_clsf_loader, eval_clsf_loader, train_clsf_sampler, eval_clsf_sampler, scaler, global_rank, local_rank, world_size, is_master, device)
        
    if is_master:
        main.writer.close()



if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--computer',
        choices=['jeanzay', 'other'], default='other',
        help='Computer used')

    parser.add_argument('--hardware',
        choices=['multi-gpu', 'mono-gpu', 'cpu'], default='cpu',
        help='Type of hardware to use')

    parser.add_argument('--precision',
        choices=['normal', 'mixed'], default='normal',
        help='Type of precision to use')

    parser.add_argument('--nb_workers', type=int,
        default=10,
        help='Number of workers')

    parser.add_argument('--dataset_name',
        choices=['cifar10', 'imagenet'], default='cifar10',
        help='Name of the dataset')

    parser.add_argument('--resnet_type',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18',
        help='Type of resnet')

    parser.add_argument('--z_dim', type=int,
        default=256,
        help='Dimension of z')

    parser.add_argument('--nb_epochs_clsf', type=int,
        default=90,
        help='Number of epochs for linear eval')

    parser.add_argument('--batch_size_clsf', type=int,
        default=256,
        help='Size of the global batch for linear eval')

    parser.add_argument('--lr_init_clsf', type=float,
        default=0.2,
        help='Initial learning rate for linear eval')

    parser.add_argument('--momentum_clsf', type=float,
        default=0.9,
        help='Momentum for linear eval')

    parser.add_argument('--weight_decay_clsf', type=float,
        default=0.0,
        help='Weight decay for linear eval')

    parser.add_argument('--checkpoint',
        default=None,
        help='Path to the checkpoint to start from')

    # Parse arguments
    args = parser.parse_args()

    # Run main fonction
    run(args)
