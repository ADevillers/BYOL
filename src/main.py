from dataset import *
from loss import *
from lars import *
from model import *
from utils import *

import argparse
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule


tracker = None
writer = None



def init():
    # Define master address and port
    import idr_torch

    # Get task information
    global_rank = idr_torch.rank
    local_rank = idr_torch.local_rank
    world_size = idr_torch.size
    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])
    n_gpu_per_node = world_size // n_nodes
    is_master = global_rank == 0
    
    # Print all task information
    print('{:>2}> Task {:>2} in {:>2} | Node {} in {} | GPU {} in {} | {}'.format(
        global_rank, global_rank, world_size, node_id, n_nodes, local_rank, n_gpu_per_node, 'Master' if is_master else '-'))
    
    # Set the device to use
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')
    
    # Init the process group
    torch.distributed.init_process_group(
        init_method='env://',
        backend='nccl',
        world_size=world_size, 
        rank=global_rank
    )

    return global_rank, local_rank, world_size, is_master, device



def train_repr(model, momentum_model, optimizer, criterion, loader, scheduler, scaler, args, device, is_master, global_rank, world_size):
    model.train()

    tracker.reset('train_loss_z')
    tracker.reset('train_loss_repr')

    tracker.reset('train_h_norm_mean')
    tracker.reset('train_h_norm_std')
    tracker.reset('train_z_norm_mean')
    tracker.reset('train_z_norm_std')
    tracker.reset('train_q_norm_mean')
    tracker.reset('train_q_norm_std')

    tracker.reset('train_m_h_norm_mean')
    tracker.reset('train_m_h_norm_std')
    tracker.reset('train_m_z_norm_mean')
    tracker.reset('train_m_z_norm_std')

    tracker.reset('train_tau')

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        (img_1, img_2), label = batch
        img_1 = img_1.to(device, non_blocking=True)
        img_2 = img_2.to(device, non_blocking=True)

        x = torch.cat([img_1, img_2], dim=0)

        with torch.cuda.amp.autocast(args.precision == 'mixed'):
            step = (tracker.get('epoch').count - 1)*len(loader) + i
            total_step = args.nb_epochs*len(loader)

            with torch.no_grad():
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    momentum_model.module.update(model.module.resnet, model.module.proj_head_inv, step, total_step)
                    tracker.add('train_tau', momentum_model.module.tau, 1)
                else:
                    momentum_model.update(model.resnet, model.proj_head_inv, step, total_step)
                    tracker.add('train_tau', momentum_model.tau, 1)

            h, z, q = model(x)

            with torch.no_grad():
                m_h, m_z = momentum_model(x)
                m_h, m_z = m_h.to(q.device), m_z.to(q.device)

        batch_size = img_1.shape[0]

        with torch.cuda.amp.autocast(args.precision == 'mixed'):
            loss_z = criterion['BYOLLoss'](m_z, q)
            loss = loss_z

        scaler.scale(loss).backward() if args.precision == 'mixed' else loss.backward()
        scaler.step(optimizer) if args.precision == 'mixed' else optimizer.step()
        if args.precision == 'mixed': scaler.update()
        if scheduler is not None: scheduler.step()

        tracker.add('train_loss_z', loss_z.detach()*batch_size/world_size, batch_size*2)
        tracker.add('train_loss_repr', loss.detach()*batch_size, batch_size*2)

        tracker.add('train_h_norm_mean', h.detach().norm(dim=1).mean(), 1)
        tracker.add('train_h_norm_std', h.detach().norm(dim=1).std(), 1)
        tracker.add('train_z_norm_mean', z.detach().norm(dim=1).mean(), 1)
        tracker.add('train_z_norm_std', z.detach().norm(dim=1).std(), 1)
        tracker.add('train_q_norm_mean', q.detach().norm(dim=1).mean(), 1)
        tracker.add('train_q_norm_std', q.detach().norm(dim=1).std(), 1)

        tracker.add('train_m_h_norm_mean', m_h.detach().norm(dim=1).mean(), 1)
        tracker.add('train_m_h_norm_std', m_h.detach().norm(dim=1).std(), 1)
        tracker.add('train_m_z_norm_mean', m_z.detach().norm(dim=1).mean(), 1)
        tracker.add('train_m_z_norm_std', m_z.detach().norm(dim=1).std(), 1)

        if is_master:
            print('\rTrain Batch: {:>3}/{:<3}...'.format(i + 1, len(loader)), end='')

    if args.hardware == 'multi-gpu':
        tracker.get('train_loss_z').reduce()    
        tracker.get('train_loss_repr').reduce()
    
    if is_master:
        writer.add_scalar('train_loss_z', tracker.get('train_loss_z').average, tracker.get('epoch').count)
        writer.add_scalar('train_loss_repr', tracker.get('train_loss_repr').average, tracker.get('epoch').count)

        writer.add_scalar('train_h_norm_mean', tracker.get('train_h_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_h_norm_std', tracker.get('train_h_norm_std').average, tracker.get('epoch').count)
        writer.add_scalar('train_z_norm_mean', tracker.get('train_z_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_z_norm_std', tracker.get('train_z_norm_std').average, tracker.get('epoch').count)
        writer.add_scalar('train_q_norm_mean', tracker.get('train_q_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_q_norm_mean', tracker.get('train_q_norm_mean').average, tracker.get('epoch').count)

        writer.add_scalar('train_m_h_norm_mean', tracker.get('train_m_h_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_m_h_norm_std', tracker.get('train_m_h_norm_std').average, tracker.get('epoch').count)
        writer.add_scalar('train_m_z_norm_mean', tracker.get('train_m_z_norm_mean').average, tracker.get('epoch').count)
        writer.add_scalar('train_m_z_norm_std', tracker.get('train_m_z_norm_std').average, tracker.get('epoch').count)

        writer.add_scalar('train_tau', tracker.get('train_tau').average, tracker.get('epoch').count)



def eval_repr(model, momentum_model, criterion, loader, args, device, is_master, global_rank, world_size):
    model.eval()

    tracker.reset('eval_loss_z')
    tracker.reset('eval_loss_repr')

    tracker.reset('eval_h_norm_mean')
    tracker.reset('eval_h_norm_std')
    tracker.reset('eval_z_norm_mean')
    tracker.reset('eval_z_norm_std')
    tracker.reset('eval_q_norm_mean')
    tracker.reset('eval_q_norm_std')

    tracker.reset('eval_m_h_norm_mean')
    tracker.reset('eval_m_h_norm_std')
    tracker.reset('eval_m_z_norm_mean')
    tracker.reset('eval_m_z_norm_std')

    with torch.no_grad():
        for i, batch in enumerate(loader):
            (img_1, img_2), label = batch
            img_1 = img_1.to(device, non_blocking=True)
            img_2 = img_2.to(device, non_blocking=True)

            x = torch.cat([img_1, img_2], dim=0)

            with torch.cuda.amp.autocast(args.precision == 'mixed'):
                h, z, q = model(x)

                m_h, m_z = momentum_model(x)
                m_h, m_z = m_h.to(q.device), m_z.to(q.device)

            batch_size = img_1.shape[0]

            with torch.cuda.amp.autocast(args.precision == 'mixed'):
                loss_z = criterion['BYOLLoss'](m_z, q)
                loss = loss_z

            tracker.add('eval_loss_z', loss_z*batch_size/world_size, batch_size*2)
            tracker.add('eval_loss_repr', loss*batch_size, batch_size*2)

            tracker.add('eval_h_norm_mean', h.norm(dim=1).mean(), 1)
            tracker.add('eval_h_norm_std', h.norm(dim=1).std(), 1)
            tracker.add('eval_z_norm_mean', z.norm(dim=1).mean(), 1)
            tracker.add('eval_z_norm_std', z.norm(dim=1).std(), 1)
            tracker.add('eval_q_norm_mean', q.norm(dim=1).mean(), 1)
            tracker.add('eval_q_norm_std', q.norm(dim=1).std(), 1)

            tracker.add('eval_m_h_norm_mean', m_h.norm(dim=1).mean(), 1)
            tracker.add('eval_m_h_norm_std', m_h.norm(dim=1).std(), 1)
            tracker.add('eval_m_z_norm_mean', m_z.norm(dim=1).mean(), 1)
            tracker.add('eval_m_z_norm_std', m_z.norm(dim=1).std(), 1)
            
            if is_master:
                print('\rEval Batch: {:>3}/{:<3}...'.format(i + 1, len(loader)), end='')

        if args.hardware == 'multi-gpu':
            tracker.get('eval_loss_z').reduce()
            tracker.get('eval_loss_repr').reduce()
        
        if is_master:
            writer.add_scalar('eval_loss_z', tracker.get('eval_loss_z').average, tracker.get('epoch').count)
            writer.add_scalar('eval_loss_repr', tracker.get('eval_loss_repr').average, tracker.get('epoch').count)

            writer.add_scalar('eval_h_norm_mean', tracker.get('eval_h_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_h_norm_std', tracker.get('eval_h_norm_std').average, tracker.get('epoch').count)
            writer.add_scalar('eval_z_norm_mean', tracker.get('eval_z_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_z_norm_std', tracker.get('eval_z_norm_std').average, tracker.get('epoch').count)
            writer.add_scalar('eval_q_norm_mean', tracker.get('eval_q_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_q_norm_std', tracker.get('eval_q_norm_std').average, tracker.get('epoch').count)

            writer.add_scalar('eval_m_h_norm_mean', tracker.get('eval_m_h_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_m_h_norm_std', tracker.get('eval_m_h_norm_std').average, tracker.get('epoch').count)
            writer.add_scalar('eval_m_z_norm_mean', tracker.get('eval_m_z_norm_mean').average, tracker.get('epoch').count)
            writer.add_scalar('eval_m_z_norm_std', tracker.get('eval_m_z_norm_std').average, tracker.get('epoch').count)



def train_clsf(model, linear, optimizer, criterion, loader, scheduler, scaler, args, device, is_master, global_rank, world_size):
    model.eval()
    linear.train()

    tracker.reset('train_loss_clsf')
    tracker.reset('train_acc_1')
    tracker.reset('train_acc_5')
    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        image_0, target = batch
        image_0 = image_0.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_size = image_0.shape[0]

        with torch.cuda.amp.autocast(args.precision == 'mixed'):
            with torch.no_grad():
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    h = model.module.resnet(image_0)
                else:
                    h = model.resnet(image_0)
            c = linear(h)

            loss = criterion(c, target)

        scaler.scale(loss).backward() if args.precision == 'mixed' else loss.backward()
        scaler.step(optimizer) if args.precision == 'mixed' else optimizer.step()
        if args.precision == 'mixed': scaler.update()
        scheduler.step()

        # Compute accuracy (top-1 and top-5)
        with torch.no_grad():
            _, pred = c.topk(5, 1, True, True)
            correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).float()

            tracker.add('train_loss_clsf', loss.detach()*batch_size, batch_size)
            tracker.add('train_acc_1', correct[:, :1].sum(), batch_size)
            tracker.add('train_acc_5', correct[:, :5].sum(), batch_size)

        if is_master:
            print('\rTrain Batch: {:>3}/{:<3}'.format(i + 1, len(loader)), end='')

    if args.hardware == 'multi-gpu':
        tracker.get('train_loss_clsf').reduce()
        tracker.get('train_acc_1').reduce()
        tracker.get('train_acc_5').reduce()
    
    if is_master:
        writer.add_scalar('curve_{}_train_loss_clsf'.format(tracker.get('epoch').count), tracker.get('train_loss_clsf').average, tracker.get('epoch_clsf').count)
        writer.add_scalar('curve_{}_train_acc_1'.format(tracker.get('epoch').count), tracker.get('train_acc_1').average, tracker.get('epoch_clsf').count)
        writer.add_scalar('curve_{}_train_acc_5'.format(tracker.get('epoch').count), tracker.get('train_acc_5').average, tracker.get('epoch_clsf').count)



def eval_clsf(model, linear, criterion, loader, args, device, is_master, global_rank, world_size):
    model.eval()
    linear.eval()

    tracker.reset('eval_loss_clsf')
    tracker.reset('eval_acc_1')
    tracker.reset('eval_acc_5')
    with torch.no_grad():
        for i, batch in enumerate(loader):
            image_0, target = batch
            image_0 = image_0.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            batch_size = image_0.shape[0]

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                h = model.module.resnet(image_0)
            else:
                h = model.resnet(image_0)
            c = linear(h)

            loss = criterion(c, target)

            # Compute accuracy (top-1 and top-5)
            _, pred = c.topk(5, 1, True, True)
            correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).float()

            tracker.add('eval_loss_clsf', loss.detach()*batch_size, batch_size)
            tracker.add('eval_acc_1', correct[:, :1].sum(), batch_size)
            tracker.add('eval_acc_5', correct[:, :5].sum(), batch_size)

            if is_master:
                print('\rTrain Batch: {:>3}/{:<3}'.format(i + 1, len(loader)), end='')

        if args.hardware == 'multi-gpu':
            tracker.get('eval_loss_clsf').reduce()
            tracker.get('eval_acc_1').reduce()
            tracker.get('eval_acc_5').reduce()
        
        if is_master:
            writer.add_scalar('curve_{}_eval_loss_clsf'.format(tracker.get('epoch').count), tracker.get('eval_loss_clsf').average, tracker.get('epoch_clsf').count)
            writer.add_scalar('curve_{}_eval_acc_1'.format(tracker.get('epoch').count), tracker.get('eval_acc_1').average, tracker.get('epoch_clsf').count)
            writer.add_scalar('curve_{}_eval_acc_5'.format(tracker.get('epoch').count), tracker.get('eval_acc_5').average, tracker.get('epoch_clsf').count)



def linear_eval(args, model, train_clsf_loader, eval_clsf_loader, train_clsf_sampler, eval_clsf_sampler, scaler, global_rank, local_rank, world_size, is_master, device):
    ######### MODEL and CO #########
    # CREATE LINEAR MODEL
    in_dim = model.module.h_dim if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.h_dim
    linear = torch.nn.Linear(in_dim, get_nbclasses(args.dataset_name)).to(device)
    if args.hardware == 'multi-gpu':
        linear = DDP(linear, device_ids=[local_rank])
    
    # CREATE CRITERION
    criterion_clsf = torch.nn.CrossEntropyLoss()

    # CREATE OPTIMIZER
    optimizer_clsf = torch.optim.SGD(
        linear.parameters(),
        lr=args.lr_init_clsf,
        momentum=args.momentum_clsf,
        weight_decay=args.weight_decay_clsf,
        nesterov=True
    )

    # CREATE SCHEDULER
    cosine_scheduler_clsf = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_clsf, args.nb_epochs_clsf*len(train_clsf_loader), eta_min=0.0001
    )


    ######### ITERATE EPOCHS CLSF #########
    tracker.reset('epoch_clsf')
    for epoch in range(args.nb_epochs_clsf):
        start = time.time()
        tracker.add('epoch_clsf', 0)

        if args.hardware == 'multi-gpu':
            train_clsf_sampler.set_epoch(epoch)
            eval_clsf_sampler.set_epoch(epoch)

        train_clsf(model, linear, optimizer_clsf, criterion_clsf, train_clsf_loader, cosine_scheduler_clsf, scaler, args, device, is_master, global_rank, world_size)
        eval_clsf(model, linear, criterion_clsf, eval_clsf_loader, args, device, is_master, global_rank, world_size)

        if is_master:
            print('\rEpoch: {:>3}/{:<3}  |  Loss Train: {:e}  |  Loss Eval: {:e}  |  Top-1 Train: {:>.5f}  |  Top-1 Eval: {:>.5f}  |  Time taken: {}'
                .format(epoch + 1, args.nb_epochs_clsf, tracker.get('train_loss_clsf').average, tracker.get('eval_loss_clsf').average, tracker.get('train_acc_1').average, tracker.get('eval_acc_1').average, time.time() - start))
        
    if is_master:
        writer.add_scalar('train_loss_clsf', tracker.get('train_loss_clsf').average, tracker.get('epoch').count)
        writer.add_scalar('train_acc_1', tracker.get('train_acc_1').average, tracker.get('epoch').count)
        writer.add_scalar('train_acc_5', tracker.get('train_acc_5').average, tracker.get('epoch').count)

        writer.add_scalar('eval_loss_clsf', tracker.get('eval_loss_clsf').average, tracker.get('epoch').count)
        writer.add_scalar('eval_acc_1', tracker.get('eval_acc_1').average, tracker.get('epoch').count)
        writer.add_scalar('eval_acc_5', tracker.get('eval_acc_5').average, tracker.get('epoch').count)



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
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # INIT TRACKER
    global tracker
    tracker = Tracker(device)

    # INIT WRITER
    if is_master:
        global writer

        if args.computer == 'jeanzay':
            log_dir = './runs/{}_{}_{}'.format(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_JOB_ID'], args.expe_name)
        else:
            expe_time = int(time.time())
            log_dir = './runs/{}_{}_{}'.format('expe', expe_time, args.expe_name)
        
        if checkpoint is not None:
            log_dir = checkpoint['log_dir']

        writer = SummaryWriter(log_dir)

        print('\nArgs:', args, '\n')

    
    ######### DATASETS #########
    # CREATE DATASETS
    train_repr_set = get_dataset(args.dataset_name, 'train_repr')
    eval_repr_set = get_dataset(args.dataset_name, 'eval_repr')
    train_clsf_set = get_dataset(args.dataset_name, 'train_clsf')
    eval_clsf_set = get_dataset(args.dataset_name, 'eval_clsf')

    # CREATE SAMPLERS
    if args.hardware == 'multi-gpu':
        train_repr_sampler = torch.utils.data.distributed.DistributedSampler(
            train_repr_set,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
        eval_repr_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_repr_set,
            num_replicas=world_size,
            rank=global_rank
        )
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
        train_repr_sampler = torch.utils.data.RandomSampler(
            train_repr_set
        )
        eval_repr_sampler = torch.utils.data.SequentialSampler(
            eval_repr_set
        )
        train_clsf_sampler = torch.utils.data.RandomSampler(
            train_clsf_set
        )
        eval_clsf_sampler = torch.utils.data.SequentialSampler(
            eval_clsf_set
        )
    
    # CREATE DATALOADERS
    train_repr_loader = torch.utils.data.DataLoader(
        train_repr_set,
        batch_size=args.batch_size//world_size,
        drop_last=True,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=train_repr_sampler
    )
    eval_repr_loader = torch.utils.data.DataLoader(
        eval_repr_set,
        batch_size=args.batch_size//world_size,
        num_workers=args.nb_workers,
        pin_memory=True,
        sampler=eval_repr_sampler
    )
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
    momentum_model = M_BYOL(model.resnet, model.proj_head_inv, args.tau_base).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        momentum_model.load_state_dict(checkpoint['momentum_model'])
    if args.hardware == 'multi-gpu':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        momentum_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(momentum_model)
        momentum_model = DDP(momentum_model, device_ids=[local_rank], find_unused_parameters=True)

    # CREATE CRITERION
    criterion_repr = {
        'BYOLLoss': BYOLLoss()
    }

    # CREATE OPTIMIZER
    wd_params, no_wd_params = get_model_params_groups(model)
    optimizer_repr = LARSW(
        [
            {'params': no_wd_params, 'weight_decay': 0.0, 'lars_weight': 0.0},
            {'params': wd_params}
        ],
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        eta=args.eta
    )
    
    if checkpoint is not None:
        optimizer_repr.load_state_dict(checkpoint['optimizer_repr'])

    # CREATE SCHEDULERS
    last_epoch = checkpoint['last_epoch'] if checkpoint is not None else -1 
    last_step = last_epoch*len(train_repr_loader) if checkpoint is not None else -1 
    warm_up_scheduler_repr = torch.optim.lr_scheduler.LinearLR(
        optimizer_repr, start_factor=1e-8, end_factor=1, total_iters=args.nb_epochs_warmup*len(train_repr_loader), last_epoch=last_step
    )

    cosine_scheduler_repr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_repr, (args.nb_epochs - args.nb_epochs_warmup)*len(train_repr_loader), last_epoch=last_step, eta_min=1e-8
    )
    if checkpoint is not None:
        warm_up_scheduler_repr.load_state_dict(checkpoint['warm_up_scheduler_repr'])
        cosine_scheduler_repr.load_state_dict(checkpoint['cosine_scheduler_repr'])

    # CREATE SCALER
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'mixed' else None


    ######### ITERATE EPOCHS #########
    tracker.reset('epoch')
    tracker.add('epoch', 0, max(0, last_epoch))
    for epoch in range(max(0, last_epoch + 1), args.nb_epochs):
        start = time.time()
        tracker.add('epoch', 0)

        if args.hardware == 'multi-gpu':
            train_repr_sampler.set_epoch(epoch)
            eval_repr_sampler.set_epoch(epoch)

        scheduler_repr = None
        if epoch < args.nb_epochs_warmup:
            scheduler_repr = warm_up_scheduler_repr
        else:
            scheduler_repr = cosine_scheduler_repr

        train_repr(model, momentum_model, optimizer_repr, criterion_repr, train_repr_loader, scheduler_repr, scaler, args, device, is_master, global_rank, world_size)
        eval_repr(model, momentum_model, criterion_repr, eval_repr_loader, args, device, is_master, global_rank, world_size)

        if is_master:
            if scheduler_repr is not None:
                writer.add_scalar('lr_repr', scheduler_repr.get_last_lr()[0], epoch)

            print('\rEpoch: {:>3}/{:<3}  |  Train loss: {:e}  |  Eval loss: {:e}  |  Time taken: {}'
                .format(epoch + 1, args.nb_epochs, tracker.get('train_loss_repr').average, tracker.get('eval_loss_repr').average, time.time() - start))

        # SAVE MODEL       
        if is_master:
            checkpoint_data = {
                'last_epoch': epoch,
                'log_dir': log_dir,
                'model': model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                'momentum_model': momentum_model.module.state_dict() if isinstance(momentum_model, torch.nn.parallel.DistributedDataParallel) else momentum_model.state_dict(),
                'optimizer_repr': optimizer_repr.state_dict(),
                'warm_up_scheduler_repr': warm_up_scheduler_repr.state_dict(),
                'cosine_scheduler_repr': cosine_scheduler_repr.state_dict()
            }

            # Save when it is time/last epoch
            if (epoch + 1)%args.save_every == 0 or (epoch + 1) == args.nb_epochs:
                if args.computer == 'jeanzay':
                    checkpoint_file = './checkpoints/{}_{}_{}.pt'.format(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_JOB_ID'], epoch)
                else:
                    checkpoint_file = './checkpoints/{}_{}_{}.pt'.format('expe', expe_time, epoch)

                torch.save(checkpoint_data, checkpoint_file)
            
            # Save in case of crash
            if args.computer == 'jeanzay':
                checkpoint_file = './checkpoints/{}_{}_{}.pt'.format(os.environ['SLURM_JOB_NAME'], os.environ['SLURM_JOB_ID'], 'even' if epoch%2 == 0 else 'odd')
            else:
                checkpoint_file = './checkpoints/{}_{}_{}.pt'.format('expe', expe_time, 'even' if epoch%2 == 0 else 'odd')
            torch.save(checkpoint_data, checkpoint_file)


        # If it is time to linear eval/last epoch
        if (epoch + 1)%args.clsf_every == 0 or (epoch + 1) == args.nb_epochs:
            linear_eval(args, model, train_clsf_loader, eval_clsf_loader, train_clsf_sampler, eval_clsf_sampler, scaler, global_rank, local_rank, world_size, is_master, device)
        
    if is_master:
        writer.close()



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

    parser.add_argument('--expe_name',
        default='default',
        help='Name of the expe')

    parser.add_argument('--dataset_name',
        choices=['cifar10', 'imagenet'], default='cifar10',
        help='Name of the dataset')

    parser.add_argument('--resnet_type',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18',
        help='Type of resnet')
    
    parser.add_argument('--nb_epochs', type=int,
        default=800,
        help='Number of epochs')
    
    parser.add_argument('--nb_epochs_warmup', type=int,
        default=10,
        help='Number of epochs for the lr warmup')

    parser.add_argument('--batch_size', type=int,
        default=512,
        help='Size of the global batch')

    parser.add_argument('--lr_init', type=float,
        default=2.0,
        help='Initial learning rate')

    parser.add_argument('--momentum', type=float,
        default=0.9,
        help='Momentum')

    parser.add_argument('--weight_decay', type=float,
        default=1e-6,
        help='Weight decay')

    parser.add_argument('--eta', type=float,
        default=1e-3,
        help='Eta LARS parameter')

    parser.add_argument('--z_dim', type=int,
        default=256,
        help='Dimension of z')

    parser.add_argument('--tau_base', type=float,
        default=0.996,
        help='Tau base for the momentum encoder')
    
    parser.add_argument('--clsf_every', type=int,
        default=100,
        help='Number of epochs between each linear eval')

    parser.add_argument('--save_every', type=int,
        default=100,
        help='Number of epochs between each checkpoint')

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
