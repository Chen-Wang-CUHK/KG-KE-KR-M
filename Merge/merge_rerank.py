#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import math

from onmt.reranker.reranker_scorer import build_reranker_scorer
from onmt.utils.logging import init_logger
from evaluation_utils import evaluate_func
from nltk.stem.porter import *

import onmt.opts

STEMMER = PorterStemmer()


def get_extracted_keys_fr_line(src_line, sel_probs_line, prob_th=0.7, ratio_th=0.3):
    ex_keys = []
    ex_probs = []
    src_line = src_line.strip().split()
    sel_probs_line = [float(prob) for prob in sel_probs_line.strip().split(' ; ')]
    if max(sel_probs_line) < prob_th:
        prob_th = max(sel_probs_line) * 0.8
    assert len(src_line) == len(sel_probs_line)
    idx = 0
    previous_idx = -1
    ex_key = []
    ex_prob = []
    while idx < len(src_line):
        if sel_probs_line[idx] >= prob_th:
            if idx == (previous_idx + 1) or previous_idx == -1:
                ex_key.append(src_line[idx])
                ex_prob.append(sel_probs_line[idx])
            else:
                ex_key = [src_line[idx]]
                ex_prob = [sel_probs_line[idx]]
            previous_idx = idx
        elif len(ex_key) != 0:
            ex_keys.append(' '.join(ex_key))
            ex_probs.append(ex_prob)

            previous_idx = -1
            ex_key = []
            ex_prob = []
        idx += 1
    if len(ex_key) != 0:
        ex_keys.append(' '.join(ex_key))
        ex_probs.append(ex_prob)

    mean_ex_probs = [round(sum(ex_prob) / len(ex_prob), 5) for ex_prob in ex_probs]
    ranked_ex_keys_pair = [(x, prob) for x, prob in
                           sorted(zip(ex_keys, mean_ex_probs), key=lambda pair: pair[1], reverse=True)]
    ranked_ex_keys = [pair[0] for pair in ranked_ex_keys_pair]
    ranked_ex_probs = [pair[1] for pair in ranked_ex_keys_pair]
    assert len(ranked_ex_keys) != 0

    return ranked_ex_keys, ranked_ex_probs


def get_extracted_keys(opt):
    src_file = open(opt.src, encoding='utf-8')
    src_lines = src_file.readlines()
    sel_probs = open(opt.sel_probs, encoding='utf-8')
    sel_probs_lines = sel_probs.readlines()

    sel_keys_out = open(opt.sel_keys_output, 'w', encoding='utf-8')
    for src_line, sel_probs_line in zip(src_lines, sel_probs_lines):
        ex_keys, ex_probs = get_extracted_keys_fr_line(src_line, sel_probs_line)
        # mul_ex_probs = [round(numpy.prod(ex_prob), 5) for ex_prob in ex_probs]
        sel_keys_out.write(' ; '.join(ex_keys) + '\n')


def main1(opt, logger):
    reranker_scorer = build_reranker_scorer(opt)

    src_file = open(opt.src, encoding='utf-8')
    src_lines = src_file.readlines()
    tgt_file = open(opt.tgt, encoding='utf-8')
    tgt_lines = tgt_file.readlines()
    assert len(src_lines) == len(tgt_lines)
    logger.info('Reranking {} ... '.format(opt.tgt))

    # generated keyphrases
    gen_scores_file = open(opt.gen_scores, encoding='utf-8')
    gen_scores_lines = gen_scores_file.readlines()
    assert len(tgt_lines) == len(gen_scores_lines)

    # retrieved keyphrases
    if opt.merge_rk_keys:
        retrieved_keys = open(opt.retrieved_keys, encoding='utf-8')
        rk_lines = retrieved_keys.readlines()
        assert len(src_lines) == len(rk_lines)

        retrieved_scores = open(opt.retrieved_scores, encoding='utf-8')
        rsc_lines = retrieved_scores.readlines()
        assert len(src_lines) == len(rsc_lines)

    # extracted key words
    if opt.merge_ex_keys:
        sel_probs = open(opt.sel_probs, encoding='utf-8')
        sel_probs_lines = sel_probs.readlines()

    # ground_truth keywords
    gt_keys = open(opt.kpg_tgt, encoding='utf-8')
    gt_keys_lines = gt_keys.readlines()
    assert len(src_lines) == len(gt_keys_lines)

    sel_keys_out = open(opt.sel_keys_output, 'w', encoding='utf-8')
    out_file = open(opt.reranked_scores_output, 'w', encoding='utf-8')
    reranked_out_file = open(opt.output, 'w', encoding='utf-8')

    report_every = int(min(len(src_lines)/10, 200))
    for i in range(len(src_lines)):
        if (i + 1) % report_every == 0:
            logger.info('{} papers complete!'.format(i + 1))
        src_line = src_lines[i]
        tgt_line = tgt_lines[i]
        gen_scores_line = gen_scores_lines[i]
        gt_keys_line = gt_keys_lines[i]

        # get the predicted keys and scores by the generator
        gen_scores = []
        expand_tgt = []
        for key, gen_sc in zip(tgt_line.strip().split(' ; '), gen_scores_line.strip().split(' ; ')):
            key = key.strip()
            if len(key) > 2:
                expand_tgt.append(key)
                gen_scores.append(float(gen_sc.strip()))
        gen_num = len(expand_tgt)

        # get the predicted keys and scores by the retriever
        rk_scores = []
        expand_rk = []
        if opt.merge_rk_keys:
            rk_line = rk_lines[i]
            rsc_line = rsc_lines[i]
            for key, rk_sc in zip(rk_line.strip().split(' <eos> '), rsc_line.strip().split(' ')):
                key = key.strip()
                if len(key) > 2:
                    expand_rk.append(key)
                    rk_scores.append(float(rk_sc.strip()))
        rk_num = len(expand_rk)

        # get the extracted keys and predicted scores by the extactor
        ex_keys = []
        ex_scores = []
        if opt.merge_ex_keys:
            sel_probs_line = sel_probs_lines[i]
            ex_keys, ex_scores = get_extracted_keys_fr_line(src_line, sel_probs_line)
            # store the extracted keys
            sel_keys_out.write(' ; '.join(ex_keys) + '\n')
        ex_num = len(ex_keys)

        expand_src = [src_line] * (gen_num + rk_num + ex_num)
        merged_tgt = expand_tgt + expand_rk + ex_keys

        # dynamically give weights to rk_scores
        rescaled_rk_scores = []
        if rk_num != 0:
            rk_lambda = (sum(gen_scores) / gen_num) / (sum(rk_scores) / rk_num)
            rescaled_rk_scores = [round(rk_sc * rk_lambda, 5) for rk_sc in rk_scores]
        # dynamically give weights to ex_scores
        rescaled_ex_scores = []
        if ex_num != 0:
            ex_lambda = ((sum(gen_scores) / gen_num) / (sum(ex_scores) / ex_num))
            rescaled_ex_scores = [round(ex_sc * ex_lambda, 5) for ex_sc in ex_scores]

        merged_scores = gen_scores + rescaled_rk_scores + rescaled_ex_scores

        scored_triplets = []
        loop_num = int(math.ceil(len(merged_tgt) / 100))
        for loop_idx in range(loop_num):
            start_idx = loop_idx * 100
            end_idx = min((loop_idx + 1) * 100, len(merged_tgt))
            scored_triplets_tmp = reranker_scorer.scoring(src_data_iter=expand_src[start_idx: end_idx],
                                                          tgt_data_iter=merged_tgt[start_idx: end_idx])
            scored_triplets = scored_triplets + scored_triplets_tmp

        reranker_scores = [round(triplet['score'], 5) for triplet in scored_triplets]
        assert len(reranker_scores) == len(merged_scores)
        rescaled_scores = [round(re_sc * mg_sc, 8) for re_sc, mg_sc in zip(reranker_scores, merged_scores)]

        # get the statistics of the gen_keys and rk_keys
        if opt.merge_with_stemmer:
            stemmed_expand_tgt = [' '.join([STEMMER.stem(w.strip()) for w in key.split()]) for key in expand_tgt]
            stemmed_rk_keys = [' '.join([STEMMER.stem(w.strip()) for w in key.split()]) for key in expand_rk]
            stemmed_ex_keys = [' '.join([STEMMER.stem(w.strip()) for w in key.split()]) for key in ex_keys]
            stemmed_mg_keys = stemmed_expand_tgt + stemmed_rk_keys + stemmed_ex_keys
        else:
            stemmed_expand_tgt = expand_tgt
            stemmed_rk_keys = expand_rk
            stemmed_ex_keys = ex_keys
            stemmed_mg_keys = stemmed_expand_tgt + stemmed_rk_keys + stemmed_ex_keys
            # stemmed_gt_keys_set = gt_keys_line.strip().split(' ; ')

        stem_map = {}
        for stemmed_key, key in zip(stemmed_mg_keys, merged_tgt):
            if stemmed_key not in stem_map:
                stem_map[stemmed_key] = key

        stemmed_gt_keys_set = set(
            [' '.join([STEMMER.stem(w.strip()) for w in key.split()]) for key in gt_keys_line.strip().split(' ; ')])

        # get the reranked results W/O merging the duplicates among gen_keys, rk_keys and ex_keys
        no_merge_sc_reranked_pair = [(x, sc) for x, sc in sorted(
            zip(stemmed_mg_keys, rescaled_scores), key=lambda pair: pair[1],
            reverse=True)]
        no_merge_sc_reranked_tgt = [x for x, _ in no_merge_sc_reranked_pair]

        no_merge_sc_reranked_gen_pair = [(x, sc) for x, sc in sorted(
            zip(stemmed_mg_keys[:gen_num], rescaled_scores[:gen_num]), key=lambda pair: pair[1],
            reverse=True)]
        assert len(no_merge_sc_reranked_gen_pair) == len(expand_tgt)

        no_merge_sc_reranked_rk_pair = []
        if opt.merge_rk_keys:
            no_merge_sc_reranked_rk_pair = [(x, sc) for x, sc in sorted(
                zip(stemmed_mg_keys[gen_num: gen_num + rk_num], rescaled_scores[gen_num: gen_num + rk_num]),
                key=lambda pair: pair[1], reverse=True)]
        assert len(no_merge_sc_reranked_rk_pair) == len(expand_rk)

        no_merge_sc_reranked_ex_pair = []
        if opt.merge_ex_keys:
            no_merge_sc_reranked_ex_pair = [(x, sc) for x, sc in sorted(
                zip(stemmed_mg_keys[gen_num + rk_num:], rescaled_scores[gen_num + rk_num:]), key=lambda pair: pair[1],
                reverse=True)]
        assert len(no_merge_sc_reranked_ex_pair) == len(ex_keys)

        # get the reranked results W/ merging the duplicates between gen_keys and rk_keys
        final_candies_set = {}
        sc_from_rk = {}
        sc_from_ex = {}
        for stemmed_mg_tgt, sc in no_merge_sc_reranked_gen_pair:
            if stemmed_mg_tgt not in final_candies_set:
                final_candies_set[stemmed_mg_tgt] = sc
                sc_from_rk[stemmed_mg_tgt] = 0.0
                sc_from_ex[stemmed_mg_tgt] = 0.0

        if opt.merge_rk_keys:
            for stemmed_mg_tgt, sc in no_merge_sc_reranked_rk_pair:
                if stemmed_mg_tgt in final_candies_set:
                    if sc_from_rk[stemmed_mg_tgt] == 0.0:
                        final_candies_set[stemmed_mg_tgt] += sc
                        sc_from_rk[stemmed_mg_tgt] = sc
                else:
                    final_candies_set[stemmed_mg_tgt] = sc
                    sc_from_rk[stemmed_mg_tgt] = sc
                    sc_from_ex[stemmed_mg_tgt] = 0.0

        if opt.merge_ex_keys:
            for stemmed_mg_tgt, sc in no_merge_sc_reranked_ex_pair:
                if stemmed_mg_tgt in final_candies_set:
                    if sc_from_ex[stemmed_mg_tgt] == 0.0:
                        final_candies_set[stemmed_mg_tgt] += sc
                        sc_from_ex[stemmed_mg_tgt] = sc
                else:
                    final_candies_set[stemmed_mg_tgt] = sc
                    sc_from_ex[stemmed_mg_tgt] = sc

        fn_reranked_tgt = [x for x in sorted(final_candies_set, key=final_candies_set.get, reverse=True)]

        # store the final scores of each keyphrase candidate
        scores_str = [str(final_candies_set[fn_key]) for fn_key in fn_reranked_tgt]
        score_line = ' ; '.join(scores_str) + '\n'
        out_file.write(score_line)
        # store the final merged and reranked keyphrase candidates
        fn_reranked_tgt = [stem_map[fn_key] for fn_key in fn_reranked_tgt]
        fn_reranked_tgt_line = ' ; '.join(fn_reranked_tgt) + '\n'
        reranked_out_file.write(fn_reranked_tgt_line)
    reranked_out_file.close()
    out_file.close()
    sel_keys_out.close()
    logger.info('{} papers complete!'.format(len(src_lines)))
    # evaluate the final predictions
    evaluate_func(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='merge_rerank.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.merge_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main1(opt, logger)
    # get_extracted_keys(opt)