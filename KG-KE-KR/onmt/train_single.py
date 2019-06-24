#!/usr/bin/env python
"""
    Training on a single process
"""
from __future__ import division

import argparse
import os
import random
import torch

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def _my_tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    sel = 0
    enc = 0
    dec = 0
    gen = 0
    for name, param in model.named_parameters():
        if 'selector' in name:
            sel += param.nelement()
        elif 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        elif 'generator' in name:
            gen += param.nelement()
        else:
            raise AssertionError("The parameters {} do not belong to any part!".format(name))

    return n_params, sel, enc, dec, gen


def training_opt_postprocessing(opt):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpuid:
        raise AssertionError("Using SRU requires -gpuid set.")

    if torch.cuda.is_available() and not opt.gpuid:
        logger.info("WARNING: You have a CUDA device, should run with -gpuid")

    if opt.gpuid:
        torch.cuda.set_device(opt.device_id)
        if opt.seed > 0:
            # this one is needed for torchtext random call (shuffled iterator)
            # in multi gpu it ensures datasets are read in the same order
            random.seed(opt.seed)
            # These ensure same initialization in multi gpu mode
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt):
    opt = training_opt_postprocessing(opt)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
    else:
        checkpoint = None
        model_opt = opt

    if opt.load_pretrained_selector_from:
        logger.info('Loading selector checkpoint from %s' % opt.load_pretrained_selector_from)
        sel_checkpoint = torch.load(opt.load_pretrained_selector_from,
                                    map_location=lambda storage, loc: storage)
    else:
        sel_checkpoint =None

    if opt.load_pretrained_s2s_generator_from:
        logger.info('Loading s2s generator checkpoint from %s' % opt.load_pretrained_s2s_generator_from)
        s2s_gen_checkpoint = torch.load(opt.load_pretrained_s2s_generator_from,
                                    map_location=lambda storage, loc: storage)
    else:
        s2s_gen_checkpoint = None

    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(lazily_load_dataset("train", opt))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, data_type, opt, checkpoint)

    # Report src/tgt features.
    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint, sel_checkpoint, s2s_gen_checkpoint)

    # Fix the pretrained selector parameters if needed
    if model_opt.fix_sel_all:
        assert opt.load_pretrained_selector_from
        assert opt.sel_lambda == 0.0
        assert not model_opt.fix_sel_classifier
        for name, param in model.named_parameters():
            if 'selector' in name:
                param.requires_grad = False
    # only fix the classifier of the selector
    if model_opt.fix_sel_classifier:
        assert opt.load_pretrained_selector_from
        assert not model_opt.fix_sel_all
        for name, param in model.named_parameters():
            if 'selector' in name and 'rnn' not in name and 'embeddings' not in name:
                param.requires_grad = False

    n_params, sel, enc, dec, gen = _my_tally_parameters(model)
    logger.info('selector: %d' % sel)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('generator: %d' % gen)
    logger.info('* number of parameters: %d' % n_params)

    print_trainable_parameters(model)

    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, model, fields, optim, data_type, model_saver=model_saver)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt), fields, opt)

    # Do training.
    trainer.train(train_iter_fct, valid_iter_fct, opt.train_steps,
                  opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


def print_trainable_parameters(model):
    n_trainable_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    logger.info('* number of trainable parameters: {}'.format(n_trainable_params))
    for name, param in model.named_parameters():
        logger.info("{} : {}, trainable = {}".format(name, param.data.shape, param.requires_grad))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
