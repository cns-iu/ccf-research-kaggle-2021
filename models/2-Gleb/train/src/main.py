import os
import argparse
import shutil

import torch
import torch.multiprocessing as mp

from logger import logger, log_init
from train import start
import utils 
from config import cfg, cfg_init


def set_gpus(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        gpus = ','.join([str(g) for g in cfg.TRAIN.GPUS])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logger.warning(f'WARNING, GPUS already set in env: CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")}')
      
@utils.cfg_frz
def parallel_init(args=None):
    if args is None:
        # TODO kinda stupid...
        logger.warning('Couldnt trace args (run in jupyter?), starting local initialization')
        class args: 
            local_rank=0
            #cfg = 'src/configs/unet.yaml'

    cfg.PARALLEL.LOCAL_RANK = args.local_rank 
    cfg.PARALLEL.IS_MASTER = cfg.PARALLEL.LOCAL_RANK == 0 
        
    torch.cuda.set_device(cfg.PARALLEL.LOCAL_RANK)
    
    if cfg.PARALLEL.DDP: 
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cfg.PARALLEL.WORLD_SIZE = torch.distributed.get_world_size()
    else:
        cfg.PARALLEL.WORLD_SIZE = 1
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def init_output_folder(cfg):
    output_folder=None
    if cfg.PARALLEL.LOCAL_RANK == 0:
        output_folder = utils.make_folders(cfg)
        shutil.copytree('src', str(output_folder/'src'))
        #if os.path.exists('/mnt/src'): shutil.copytree('/mnt/src', str(output_folder/'src'))
    return output_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--cfg", default='src/configs/unet.yaml', type=str)
    args = parser.parse_args()
    return args

if __name__  == "__main__":
    """
        I know this is main, but actual start happens to be in src/starter(.sh)
        This is because of torch DDP https://pytorch.org/docs/stable/notes/ddp.html ,
        which is multiprocess (several GPUs), and even multinode (several servers) setup
    """
    args = parse_args()
    cfg_init(args.cfg)
    set_gpus(cfg)
    parallel_init(args)

    if cfg.PARALLEL.IS_MASTER: logger.debug('\n' + cfg.dump(indent=4))
    output_folder = init_output_folder(cfg)
    log_init(output_folder)
    start(cfg, output_folder)
