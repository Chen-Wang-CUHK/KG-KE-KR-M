#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import argparse
import codecs
import os
import math

import torch

from itertools import count
from onmt.utils.misc import tile
from evaluation_utils import evaluate_func

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts


def build_reranker_scorer(opt, logger=None):
    # out_file = codecs.open(opt.output, 'w+', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    reranker_scorer = ReRankerScorer(opt, model_opt, model, fields, logger=logger)

    return reranker_scorer


class ReRankerScorer(object):

    def __init__(self, opt, model_opt, model, fields, out_file=None, logger=None):
        self.logger = logger
        self.gpu = opt.gpu
        self.cuda = opt.gpu > -1

        self.opt = opt
        self.model_opt = model_opt
        self.model = model
        self.fields = fields
        self.out_file = out_file

    def scoring(self, src_data_path=None, src_data_iter=None, tgt_data_path=None, tgt_data_iter=None, batch_size=32):
        if src_data_iter is not None:
            batch_size = len(src_data_iter)
        assert batch_size != 0
        data = inputters.build_dataset(self.fields,
                                       'text',
                                       src_path=src_data_path,
                                       src_data_iter=src_data_iter,
                                       tgt_path=tgt_data_path,
                                       tgt_data_iter=tgt_data_iter,
                                       use_filter_pred=False,
                                       dynamic_dict=False)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        scored_triplets = []
        for batch in data_iter:
            src = inputters.make_features(batch, 'src', 'text')  # [src_len, batch_size, num_features]
            _, src_lengths = batch.src

            tgt = inputters.make_features(batch, 'tgt', 'text')  # [tgt_len, batch_size, num_features]
            _, tgt_lengths = batch.tgt

            logits, probs = self.model(src, tgt, src_lengths, tgt_lengths)

            # Sorting
            inds, perm = torch.sort(batch.indices.data)

            # orig_src = batch.src[0].data.index_select(1, perm)
            # orig_tgt = batch.tgt[0].data.index_select(1, perm)
            orig_probs = probs.index_select(0, perm)

            for b in range(batch.batch_size):
                src_raw = data.examples[inds[b]].src
                tgt_raw = data.examples[inds[b]].tgt
                final_score = orig_probs[b].data.item()
                scored_triplets.append({'src': src_raw, 'tgt': tgt_raw, 'score': final_score})
                # if final_score > 0.5:
                #     print('=' * 30)
                #     print('src: {}'.format(' '.join(src_raw)))
                #     print('tgt: {}; score: {}'.format(' '.join(tgt_raw), final_score))
                #     print('=' * 30)

        return scored_triplets

