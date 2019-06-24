import os
import torch
import torch.nn as nn

import onmt.inputters

from collections import deque
from onmt.utils.logging import logger


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ReRankerModelSaver(opt.save_model,
                                     model,
                                     model_opt,
                                     fields,
                                     optim,
                                     opt.save_checkpoint_steps,
                                     opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint
        self.save_checkpoint_steps = save_checkpoint_steps

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def maybe_save(self, step, stats=None):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if step % self.save_checkpoint_steps != 0:
            return
        # changed for KE_KG
        chkpt, chkpt_name = self._save(step, stats)

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step, stats=None):
        """ Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """
        Simple model saver to filesystem
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=0):
        super(ModelSaver, self).__init__(
            base_path, model, model_opt, fields, optim,
            save_checkpoint_steps, keep_checkpoint)

    def _save(self, step, stats=None):
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.inputters.save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            'optim': self.optim,
        }

        if not stats:
            checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        else:
            checkpoint_path = '%s_genPPL_%6.3f_genAcc_%4.2f_step_%d.pt' \
                              % (self.base_path, stats.gen_ppl(), stats.gen_accuracy(), step)
        logger.info("Saving s2s generator checkpoint " + checkpoint_path)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)


class ReRankerModelSaver(ModelSaverBase):
    """
        Simple model saver to filesystem
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=0):
        super(ReRankerModelSaver, self).__init__(
            base_path, model, model_opt, fields, optim,
            save_checkpoint_steps, keep_checkpoint)

    def _save(self, step, stats=None):
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()
        # TODO: remove this line?
        model_state_dict = {k: v for k, v in model_state_dict.items()}
        checkpoint = {
            'reranker': model_state_dict,
            'vocab': onmt.inputters.save_fields_to_vocab(self.fields),
            'opt': self.model_opt,
            'optim': self.optim,
        }
        # 'vocab': onmt.inputters.save_fields_to_vocab(self.fields),
        # 'opt': self.model_opt,
        # 'optim': self.optim,
        if not stats:
            checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        else:
            checkpoint_path = '%s_BCELoss_%5.4f_total_acc_%5.4f_posi_acc_%5.4f_neg_acc_%5.4f_step_%d.pt' \
                              % (self.base_path, stats.ave_loss(), stats.total_acc(), stats.posi_acc(), stats.neg_acc(), step)
        logger.info("Saving reranker checkpoint " + checkpoint_path)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)