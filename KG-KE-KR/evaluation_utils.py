import math
import string
import re
import json
import argparse
import time
import numpy as np
from nltk.stem.porter import *
from data_utils import process_keyphrase, get_tokens


def evaluate_func(opts, do_stem=True):
    """
    calculate the macro-averaged precesion, recall and F1 score
    """
    context_file = open(opts.kpg_context, encoding='utf-8')
    context_lines = context_file.readlines()

    target_file = open(opts.kpg_tgt, encoding='utf-8')
    target_lines = target_file.readlines()

    preds_file = open(opts.output, encoding='utf-8')
    preds_lines = preds_file.readlines()

    # the number of examples should be the same
    assert len(context_lines) == len(preds_lines)
    assert len(preds_lines) == len(target_lines)

    stemmer = PorterStemmer()
    num_groundtruth = 0
    num_present_groundtruth = 0
    num_absent_groundtruth = 0
    min_num_present_preds = 1000
    min_num_absent_preds = 1000
    ave_num_present_preds = 0
    ave_num_absent_preds = 0

    macro_metrics = {'total': [], 'present': [], 'absent': []}
    cnt = 1
    present_correctly_matched_at = {'5': [], '10': [], '15': [], '50': []}
    absent_correctly_matched_at = {'5': [], '10': [], '15': [], '50': []}
    total_correctly_matched_at = {'5': [], '10': [], '15': [], '50': []}
    present_target_lens_list = []
    absent_target_lens_list = []
    total_target_lens_list = []
    for context, targets, preds in zip(context_lines, target_lines, preds_lines):
        if cnt % 1000 == 0:
            print(time.strftime('%H:%M:%S') + ': {} papers evaluation complete!'.format(cnt))
        # preprocess predictions and targets to a list ['key1a key1b', 'key2a key2b']
        targets = process_keyphrase(targets.strip(), limit_num=False, fine_grad=True)
        preds = preds.replace(opts.splitter, ';')
        preds = process_keyphrase(preds.strip(), limit_num=False, fine_grad=True)
        # preprocess context in a fine-gradularity: [word1, word2,..., wordk,...]
        context = ' '.join(get_tokens(context, fine_grad=True))

        # stem words in context, target, pred, if needed
        if do_stem:
            context = ' '.join([stemmer.stem(w) for w in context.strip().split()])
            # the gold keyphrases of SemEval testing dataset are already stemmed
            if 'semeval' in opts.kpg_tgt.lower():
                targets = [' '.join([w for w in keyphrase.split()]) for keyphrase in targets]
            else:
                targets = [' '.join([stemmer.stem(w) for w in keyphrase.split()]) for keyphrase in targets]
            preds = [' '.join([stemmer.stem(w) for w in keyphrase.split()]) for keyphrase in preds]
        else:
            context = context.strip()

        if opts.filter_dot_comma_unk:
            targets = [keyphrase for keyphrase in targets if ',' not in keyphrase and '.' not in keyphrase and '<unk>' not in keyphrase]
            preds = [keyphrase for keyphrase in preds if ',' not in keyphrase and '.' not in keyphrase and '<unk>' not in keyphrase]

        # get the present_tgt_keyphrase, absent_tgt_keyphrase
        present_tgt_set = set()
        absent_tgt_set = set()
        total_tgt_set = set(targets)
        context_list = context.split()

        for tgt in targets:
            if opts.match_method == 'word_match':
                tgt_list = tgt.split()
                match = in_context2(context_list, tgt_list)
            else:
                match = tgt in context
            if match:
                present_tgt_set.add(tgt)
            else:
                absent_tgt_set.add(tgt)

        present_preds = []
        present_preds_set = set()
        absent_preds = []
        absent_preds_set = set()

        total_preds = []

        single_word_maxnum = opts.single_word_maxnum
        # split to present and absent predictions and also delete the repeated predictions
        for pred in preds:
            # # only keep single_word_maxnum single word keyphrase
            # single_word_maxnum = -1 means we keep all the single word phrase
            if single_word_maxnum != -1 and len(pred.split()) == 1:
                if single_word_maxnum > 0:
                    single_word_maxnum -= 1
                else:
                    continue
            if opts.match_method == 'word_match':
                match = in_context2(context_list, pred.split())
            else:
                match = pred in context
            if match:
                if pred not in present_preds_set:
                    total_preds.append(pred)
                    present_preds.append(pred)
                    present_preds_set.add(pred)
            else:
                if pred not in absent_preds_set:
                    total_preds.append(pred)
                    absent_preds.append(pred)
                    absent_preds_set.add(pred)

        # store the nums
        present_target_lens_list.append(len(present_tgt_set))
        absent_target_lens_list.append(len(absent_tgt_set))
        total_target_lens_list.append(len(total_tgt_set))
        num_groundtruth += len(targets)
        num_present_groundtruth += len(present_tgt_set)
        num_absent_groundtruth += len(absent_tgt_set)

        if len(present_preds_set) < min_num_present_preds:
            min_num_present_preds = len(present_preds_set)
        if len(absent_preds_set) < min_num_absent_preds:
            min_num_absent_preds = len(absent_preds_set)

        ave_num_present_preds += len(present_preds_set)
        ave_num_absent_preds += len(absent_preds_set)

        # get the correctly_matched
        total_correctly_matched = [1 if total_pred in total_tgt_set else 0 for total_pred in total_preds]
        # get the total_correctly_matched_at
        for at_key in total_correctly_matched_at:
            total_correctly_matched_at[at_key].append(total_correctly_matched[:int(at_key)])

        # get the correctly_matched
        present_correctly_matched = [1 if present_pred in present_tgt_set else 0 for present_pred in present_preds]
        # get the present_correctly_matched_at
        for at_key in present_correctly_matched_at:
            present_correctly_matched_at[at_key].append(present_correctly_matched[:int(at_key)])

        absent_correctly_matched = [1 if absent_pred in absent_tgt_set else 0 for absent_pred in absent_preds]
        # get the present_correctly_matched_at
        for at_key in absent_correctly_matched_at:
            absent_correctly_matched_at[at_key].append(absent_correctly_matched[:int(at_key)])

        # macro metric calculating
        macro_metrics['total'].append(
            macro_metric_fc(total_tgt_set, total_correctly_matched))
        macro_metrics['present'].append(
            macro_metric_fc(present_tgt_set, present_correctly_matched))

        macro_metrics['absent'].append(
            macro_metric_fc(absent_tgt_set, absent_correctly_matched))

        cnt += 1

    # compute the corpus evaluation
    print('#(Ground-truth Keyphrase)=%d' % num_groundtruth)
    print('#(Present Ground-truth Keyphrase)=%d' % num_present_groundtruth)
    print('#(Absent Ground-truth Keyphrase)=%d' % num_absent_groundtruth)

    print('#(Total Num of Present Preds Per Example)=%d' % ave_num_present_preds)
    print('#(Total Num of Absent Preds Per Example)=%d' % ave_num_absent_preds)
    print('#(Ave Num of Present Preds Per Example)=%d' % (ave_num_present_preds / (cnt - 1)))
    print('#(Ave Num of Absent Preds Per Example)=%d' % (ave_num_absent_preds / (cnt - 1)))
    print('#(Min Num of Present Preds Per Example)=%d' % min_num_present_preds)
    print('#(Min Num of Absent Preds Per Example)=%d' % min_num_absent_preds)

    # calculate and print the MAP metrics, some code are borrowed from the internet
    map_score_fc(total_correctly_matched_at, total_target_lens_list, keyphrase_type='total')
    map_score_fc(present_correctly_matched_at, present_target_lens_list, keyphrase_type='present')
    map_score_fc(absent_correctly_matched_at, absent_target_lens_list, keyphrase_type='absent')

    # calculate and print the Micro and Macro averaged F1, P, R metrics
    micro_ave_fc(macro_metrics['total'], keyphrase_type='total')
    micro_ave_fc(macro_metrics['present'], keyphrase_type='present')
    micro_ave_fc(macro_metrics['absent'], keyphrase_type='absent')


def in_context2(context_list, tgt_list):
    match = False
    for c_idx in range(len(context_list) - len(tgt_list) + 1):
        context_piece = ' '.join(context_list[c_idx: c_idx + len(tgt_list)])
        tgt_piece = ' '.join(tgt_list)
        if context_piece == tgt_piece:
            match = True
            break
    return match


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r, target_num=None):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    if target_num:
        return np.sum(out)*1.0/target_num
    else:
        return np.mean(out)


def mean_average_precision(rs, target_nums_list):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    if target_nums_list:
        return np.mean([average_precision(r, target_num) for r, target_num in zip(rs, target_nums_list)])
    else:
        return np.mean([average_precision(r) for r in rs])


def map_score_fc(correctly_matched_at, target_lens_list=None, keyphrase_type=''):
    assert keyphrase_type != ''
    print('\nBegin' + '=' * 20 + keyphrase_type + '=' * 20 + 'Begin')
    map_5 = mean_average_precision(correctly_matched_at['5'], target_lens_list)
    map_10 = mean_average_precision(correctly_matched_at['10'], target_lens_list)
    map_15 = mean_average_precision(correctly_matched_at['15'], target_lens_list)
    map_50 = mean_average_precision(correctly_matched_at['50'], target_lens_list)

    output_str = 'MAP_%s:\t\t@5=%f, @10=%f, @15=%f, @50=%f' % (keyphrase_type, map_5, map_10, map_15, map_50)
    print(output_str)
    print('End' + '=' * 20 + keyphrase_type + '=' * 20 + 'End')


def macro_metric_fc(tgt_set, correctly_matched):
    metric_dict = {}
    for number_to_predict in [5, 10, 15, 50]:
        metric_dict['target_number'] = len(tgt_set)
        metric_dict['prediction_number'] = len(correctly_matched)
        metric_dict['correct_number@%d' % number_to_predict] = sum(correctly_matched[:number_to_predict])

        metric_dict['p@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict])) / float(
            number_to_predict)

        if len(tgt_set) != 0:
            metric_dict['r@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict])) / float(
                len(tgt_set))
        else:
            metric_dict['r@%d' % number_to_predict] = 0

        if metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict] != 0:
            metric_dict['f1@%d' % number_to_predict] = 2 * metric_dict['p@%d' % number_to_predict] * metric_dict[
                'r@%d' % number_to_predict] / float(
                metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict])
        else:
            metric_dict['f1@%d' % number_to_predict] = 0
    return metric_dict


def micro_ave_fc(macro_metrics, keyphrase_type='total'):
    print('\nBegin' + '='*20 + keyphrase_type + '='*20 + 'Begin')
    real_test_size = len(macro_metrics)
    overall_score = {}
    for k in [5, 10, 15, 50]:
        correct_number = sum([m['correct_number@%d' % k] for m in macro_metrics])
        overall_target_number = sum([m['target_number'] for m in macro_metrics])
        overall_prediction_number = sum([min(m['prediction_number'], k) for m in macro_metrics])

        # Compute the Macro Measures, by averaging the macro-score of each prediction
        overall_score['p@%d' % k] = float(sum([m['p@%d' % k] for m in macro_metrics])) / float(real_test_size)
        overall_score['r@%d' % k] = float(sum([m['r@%d' % k] for m in macro_metrics])) / float(real_test_size)
        overall_score['f1@%d' % k] = float(sum([m['f1@%d' % k] for m in macro_metrics])) / float(real_test_size)

        # Print basic statistics
        output_str = 'Overall - valid testing data=%d, Number of Target=%d/%d, Number of Prediction=%d, Number of Correct=%d' % (
            real_test_size,
            overall_target_number, overall_target_number,
            overall_prediction_number, correct_number
        )
        print(output_str)
        # Print Macro-average performance
        output_str = 'Macro_%s_%d:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
            keyphrase_type, k,
            k, overall_score['p@%d' % k],
            k, overall_score['r@%d' % k],
            k, overall_score['f1@%d' % k]
        )
        print(output_str)

        # Print Micro-average performance
        overall_score['micro_p@%d' % k] = correct_number / float(overall_prediction_number) if overall_prediction_number != 0 else 0
        overall_score['micro_r@%d' % k] = correct_number / float(overall_target_number) if overall_prediction_number != 0 else 0
        if overall_score['micro_p@%d' % k] + overall_score['micro_r@%d' % k] > 0:
            overall_score['micro_f1@%d' % k] = 2 * overall_score['micro_p@%d' % k] * overall_score[
                'micro_r@%d' % k] / float(overall_score['micro_p@%d' % k] + overall_score['micro_r@%d' % k])
        else:
            overall_score['micro_f1@%d' % k] = 0

        output_str = 'Micro_%s_%d:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
            keyphrase_type, k,
            k, overall_score['micro_p@%d' % k],
            k, overall_score['micro_r@%d' % k],
            k, overall_score['micro_f1@%d' % k]
        )
        print(output_str)
        print('End' + '=' * 20 + keyphrase_type + '=' * 20 + 'End')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('-single_word_maxnum', type=int, default=-1)
    parser.add_argument('-filter_dot_comma_unk', type=bool, default=True)
    parser.add_argument('-match_method', type=str, default='word_match',
                        choices=['str_match', 'word_match'])
    parser.add_argument('-splitter', type=str, default=' ; ')

    parser.add_argument('-kpg_context', type=str,
                        default='data\\full_data\\word_kp20k_testing_context.txt')
    parser.add_argument('-kpg_tgt', type=str,
                        default='data\\full_data\\kp20k_testing_keyword.txt')
    parser.add_argument('-output', type=str,
                        default='log\\RuiMeng_translation_log\\RuiMeng_merged_predictions.txt')
    opts = parser.parse_args()

    evaluate_func(opts=opts)


