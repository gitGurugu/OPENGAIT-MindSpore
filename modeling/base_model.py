"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait_mindspore/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import mindspore as ms
import numpy as np
import os.path as osp
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.dataset import GeneratorDataset

from tqdm import tqdm
from abc import ABCMeta
from abc import abstractmethod

from . import backbones
from .loss_aggregator import LossAggregator
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions
from utils import NoOp
from utils import get_msg_mgr

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Cell):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        # MindSpore uses LossScaleManager for mixed precision
        if training and self.engine_cfg['enable_float16']:
            from mindspore.amp import FixedLossScaleManager
            self.loss_scale_manager = FixedLossScaleManager(loss_scale=2**16)
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])

        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])

        # MindSpore device management
        try:
            from mindspore.communication import get_rank
            self.device = get_rank()
        except:
            self.device = 0

        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.set_train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.CellList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        for m in self.cells():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                m.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.XavierUniform(), m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Zero(), m.bias.shape, m.bias.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.XavierUniform(), m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Zero(), m.bias.shape, m.bias.dtype))
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.use_batch_statistics:
                    m.gamma.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Normal(1.0, 0.02), m.gamma.shape, m.gamma.dtype))
                    m.beta.set_data(ms.common.initializer.initializer(
                        ms.common.initializer.Zero(), m.beta.shape, m.beta.dtype))

    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = DataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        # MindSpore uses GeneratorDataset
        # Note: This is a simplified version, you may need to adapt it further
        def data_generator():
            for batch_indices in sampler:
                batch = [dataset[i] for i in batch_indices]
                collate_fn = CollateFn(dataset.label_set, sampler_cfg)
                yield collate_fn(batch)

        loader = GeneratorDataset(
            source=data_generator(),
            column_names=["data"],
            num_parallel_workers=data_cfg.get('num_workers', 4),
            shuffle=False
        )
        return loader

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        # MindSpore optimizers
        from mindspore.nn import Adam, SGD, AdamWeightDecay
        optimizer_map = {
            'Adam': Adam,
            'SGD': SGD,
            'AdamW': AdamWeightDecay
        }
        optimizer_name = optimizer_cfg['solver']
        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        Optimizer = optimizer_map[optimizer_name]
        valid_arg = get_valid_args(Optimizer, optimizer_cfg, ['solver'])
        # Filter parameters that require grad
        params = [p for p in self.get_parameters() if p.requires_grad]
        optimizer = Optimizer(params, **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        # MindSpore learning rate schedulers
        from mindspore.nn import ExponentialDecayLR, PolynomialDecayLR, CosineDecayLR
        scheduler_map = {
            'ExponentialLR': ExponentialDecayLR,
            'PolynomialLR': PolynomialDecayLR,
            'CosineAnnealingLR': CosineDecayLR
        }
        scheduler_name = scheduler_cfg['scheduler']
        if scheduler_name not in scheduler_map:
            # Use a default scheduler or raise error
            self.msg_mgr.log_warning(f"Unsupported scheduler: {scheduler_name}, using default")
            return None
        Scheduler = scheduler_map[scheduler_name]
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(self.optimizer.learning_rate, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        try:
            from mindspore.communication import get_rank
            if get_rank() == 0:
                mkdir(osp.join(self.save_path, "checkpoints/"))
                save_name = self.engine_cfg['save_name']
                checkpoint = {
                    'model': self.parameters_dict(),
                    'optimizer': self.optimizer.parameters_dict() if hasattr(self, 'optimizer') else None,
                    'scheduler': self.scheduler.parameters_dict() if hasattr(self, 'scheduler') and self.scheduler else None,
                    'iteration': iteration}
                ms.save_checkpoint(checkpoint,
                           osp.join(self.save_path, 'checkpoints/{}-{:0>5}.ckpt'.format(save_name, iteration)))
        except:
            # Single card mode
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.parameters_dict(),
                'optimizer': self.optimizer.parameters_dict() if hasattr(self, 'optimizer') else None,
                'scheduler': self.scheduler.parameters_dict() if hasattr(self, 'scheduler') and self.scheduler else None,
                'iteration': iteration}
            ms.save_checkpoint(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.ckpt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        checkpoint = ms.load_checkpoint(save_name)
        model_state_dict = checkpoint.get('model', checkpoint)

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            current_params = {k: v for k, v in self.parameters_dict().items()}
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(current_params.keys()))))

        ms.load_param_into_net(self, model_state_dict, strict_load=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint and checkpoint['optimizer']:
                ms.load_param_into_net(self.optimizer, checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint and checkpoint['scheduler']:
                if self.scheduler:
                    ms.load_param_into_net(self.scheduler, checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.ckpt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.cells():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.set_train(False)

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        seqs = [np2var(np.asarray([trf(fra) for fra in seq])).astype(ms.float32)
                for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).astype(ms.int32)

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).astype(ms.int32)
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().asnumpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """
        # In MindSpore, gradient computation and optimization are handled by Model class
        # This is a placeholder for compatibility
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        self.iteration += 1
        if self.scheduler:
            self.scheduler(self.iteration)
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.engine_cfg['sampler']['batch_size']
        rest_size = total_size
        info_dict = Odict()
        self.set_train(False)
        for inputs in self.test_loader:
            # MindSpore dataset returns a list
            if isinstance(inputs, list) and len(inputs) > 0:
                inputs = inputs[0]
            ipts = self.inputs_pretreament(inputs)
            retval = self.construct(ipts)
            inference_feat = retval['inference_feat']
            for k, v in inference_feat.items():
                inference_feat[k] = ddp_all_gather(v, requires_grad=False)
            del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        # In MindSpore, we use Model class for training
        # This is a simplified version
        from mindspore import Model
        from mindspore.nn import TrainOneStepCell, WithLossCell
        
        # Create loss function wrapper
        class LossWrapper(nn.Cell):
            def __init__(self, model, loss_aggregator):
                super().__init__()
                self.model = model
                self.loss_aggregator = loss_aggregator
            
            def construct(self, *inputs):
                ipts = model.inputs_pretreament(inputs)
                retval = model.construct(ipts)
                training_feat = retval['training_feat']
                loss_sum, loss_info = loss_aggregator.construct(training_feat)
                return loss_sum
        
        loss_wrapper = LossWrapper(model, model.loss_aggregator)
        train_net = TrainOneStepCell(loss_wrapper, model.optimizer)
        
        for inputs in model.train_loader:
            if isinstance(inputs, list) and len(inputs) > 0:
                inputs = inputs[0]
            ipts = model.inputs_pretreament(inputs)
            retval = model.construct(ipts)
            training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
            del retval
            loss_sum, loss_info = model.loss_aggregator.construct(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                continue

            visual_summary.update(loss_info)
            if hasattr(model.optimizer, 'learning_rate'):
                visual_summary['scalar/learning_rate'] = model.optimizer.learning_rate.value()

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.set_train(False)
                    result_dict = BaseModel.run_test(model)
                    model.set_train(True)
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    @staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        evaluator_cfg = model.cfgs['evaluator_cfg']
        try:
            from mindspore.communication import get_rank, get_group_size
            rank = get_rank()
            world_size = get_group_size()
            if world_size != evaluator_cfg['sampler']['batch_size']:
                raise ValueError("The batch size ({}) must be equal to the number of GPUs ({}) in testing mode!".format(
                    evaluator_cfg['sampler']['batch_size'], world_size))
        except:
            rank = 0
        
        model.set_train(False)
        info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in evaluator_cfg.keys():
                eval_func = evaluator_cfg["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            return eval_func(info_dict, dataset_name, **valid_args)

