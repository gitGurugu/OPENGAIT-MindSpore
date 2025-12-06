import os
import argparse
import mindspore as ms
from mindspore import context
from mindspore.communication import init, get_rank, get_group_size
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait (MindSpore version).')
parser.add_argument('--local_rank', type=int, default=0,
                    help="local rank for distributed training")
parser.add_argument('--cfgs', type=str,
                    default='configs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--device_target', type=str, default='GPU',
                    choices=['GPU', 'Ascend', 'CPU'], help="device target")
parser.add_argument('--device_id', type=int, default=0, help="device id")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    try:
        rank = get_rank()
        init_seeds(rank)
    except:
        init_seeds(0)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    
    # MindSpore handles SyncBatchNorm differently
    if training and cfgs['trainer_cfg'].get('sync_BN', False):
        msg_mgr.log_warning("SyncBatchNorm is not fully supported in MindSpore, using regular BatchNorm")
    
    if cfgs['trainer_cfg'].get('fix_BN', False):
        model.fix_BN()
    
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    # Initialize MindSpore context
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target, device_id=opt.device_id)
    
    # Initialize distributed training if needed
    try:
        init()
        world_size = get_group_size()
        rank = get_rank()
        context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, 
                                         gradients_mean=True)
        msg_mgr = get_msg_mgr()
        if rank == 0:
            msg_mgr.log_info(f"Distributed training initialized: world_size={world_size}")
    except:
        # Single card mode
        pass
    
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)

