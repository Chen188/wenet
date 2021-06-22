# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch

####### binc changed to smddp #######
# import torch.distributed as dist

import smdistributed.dataparallel.torch.distributed as dist

#####################################

import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from wenet.utils.executor import Executor
from wenet.utils.scheduler import WarmupLR


# Import SMDataParallel modules for PyTorch.
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP

# SMDataParallel: Initialize
if not dist.is_initialized():
    dist.init_process_group()  # binc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')

    args = parser.parse_args()
    
    current_host= os.environ['SM_CURRENT_HOST']

    logging.basicConfig(level=logging.DEBUG,
                        format=current_host+': %(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    

    # Set random seed
    torch.manual_seed(777)
    print("args: \n", args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    distributed = dist.get_world_size() > 1  # binc
    local_rank = dist.get_local_rank()
    if distributed:
        # SMDataParallel: Pin each GPU to a single SMDataParallel process.
        torch.cuda.set_device(local_rank)
    
    
    raw_wav = configs['raw_wav']

    train_collate_func = CollateFunc(**configs['collate_conf'],
                                     raw_wav=raw_wav)

    cv_collate_conf = copy.deepcopy(configs['collate_conf'])
    # no augmenation on cv set
    cv_collate_conf['spec_aug'] = False
    cv_collate_conf['spec_sub'] = False
    if raw_wav:
        cv_collate_conf['feature_dither'] = 0.0
        cv_collate_conf['speed_perturb'] = False
        cv_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    cv_collate_func = CollateFunc(**cv_collate_conf, raw_wav=raw_wav)

    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = AudioDataset(args.train_data,
                                 **dataset_conf,
                                 raw_wav=raw_wav)
    
    cv_dataset = AudioDataset(args.cv_data, **dataset_conf, raw_wav=raw_wav)

    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(local_rank))
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank()) # binc
        cv_sampler = torch.utils.data.distributed.DistributedSampler(
            cv_dataset, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank()) # binc
    else:
        train_sampler = None
        cv_sampler = None

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=train_collate_func,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   pin_memory=args.pin_memory,
                                   batch_size=1,
                                   num_workers=args.num_workers)
    cv_data_loader = DataLoader(cv_dataset,
                                collate_fn=cv_collate_func,
                                sampler=cv_sampler,
                                shuffle=False,
                                batch_size=1,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers)

    if raw_wav:
        input_dim = configs['collate_conf']['feature_extraction_conf'][
            'mel_bins']
    else:
        input_dim = train_dataset.input_dim
    vocab_size = train_dataset.output_dim

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = raw_wav
    
    # if args.rank == 0:
    if dist.get_rank() == 0:  # binc
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_asr_model(configs)
    device = torch.device('cuda')
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
#     script_model = torch.jit.script(model)
#     script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    
    writer = None
    # if args.rank == 0:
    if dist.get_rank() == 0:  # binc
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    logging.debug('before if distributed:')
    if distributed:
        assert (torch.cuda.is_available())
        #########
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, find_unused_parameters=True)

        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)  # binc
        #########
    else:
        use_cuda = local_rank >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    logging.debug('before optimizer')
    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    final_epoch = None
    # configs['rank'] = args.rank
    configs['rank'] = dist.get_rank()  # binc
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp
    # if start_epoch == 0 and args.rank == 0:
    if start_epoch == 0 and dist.get_rank() == 0:   # binc
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    logging.debug('before epoch in range')
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs, scaler)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                configs)
#         if args.world_size > 1:
        if dist.get_world_size() > 1: # binc
            # all_reduce expected a sequence parameter, so we use [num_seen_utts].
            num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
            # the default operator in all_reduce function is sum.
            dist.all_reduce(num_seen_utts)
            total_loss = torch.Tensor([total_loss]).to(device)
            dist.all_reduce(total_loss)
            cv_loss = total_loss[0] / num_seen_utts[0]
            cv_loss = cv_loss.item()
        else:
            cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
#         if args.rank == 0:
        if dist.get_rank() == 0:   # binc
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
            writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
        final_epoch = epoch

#     if final_epoch is not None and args.rank == 0:
    if final_epoch is not None and dist.get_rank() == 0:   # binc
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
