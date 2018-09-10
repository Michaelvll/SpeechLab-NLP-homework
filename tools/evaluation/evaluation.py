#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:52:11 2018

@author: wang
"""

import os

import utils.data_reader as data_reader
import utils.acc as acc

prex_path = os.getcwd()
data_root = os.path.join(prex_path, 'data')

slot_tag_vocab_dir = os.path.join(data_root, 'lab')
intent_tag_vocab_dir = os.path.join(data_root, 'lab_unali')
valid_data_dir = os.path.join(data_root, 'valid')

slot_tag_to_idx, idx_to_slot_tag = data_reader.read_vocab_file(
    slot_tag_vocab_dir, no_pad=False, no_unk=False)
intent_tag_to_idx, idx_to_intent_tag = data_reader.read_vocab_file(
    intent_tag_vocab_dir, no_pad=True, no_unk=True)
valid_slot_tags, valid_intent_tags = data_reader.read_seqtag_data(
    valid_data_dir, slot_tag_to_idx, intent_tag_to_idx)


def decode(pred_path):
    pred_slot_tags, pred_intent_tags = data_reader.read_seqtag_data(
        pred_path, slot_tag_to_idx, intent_tag_to_idx)

    TP_1, FP_1, FN_1, TN_1 = 0.0, 0.0, 0.0, 0.0
    TP_2, FP_2, FN_2, TN_2 = 0.0, 0.0, 0.0, 0.0

    for idx, pred_line in enumerate(pred_slot_tags['data']):
        pred_seq = [idx_to_slot_tag[item] for item in pred_line]
        lab_seq = [idx_to_slot_tag[item]
                   for item in valid_slot_tags['data'][idx]]

        pred_chunks = acc.get_chunks(['O']+pred_seq+['O'])
        label_chunks = acc.get_chunks(['O']+lab_seq+['O'])
        for pred_chunk in pred_chunks:
            if pred_chunk in label_chunks:
                TP_1 += 1
            else:
                FP_1 += 1
        for label_chunk in label_chunks:
            if label_chunk not in pred_chunks:
                FN_1 += 1

    for idx, pred_line in enumerate(pred_intent_tags['data']):
        pred_seq = [0]*len(intent_tag_to_idx)
        lab_seq = [0]*len(intent_tag_to_idx)

        for item in pred_line:
            pred_seq[item] = 1
        for item in valid_intent_tags['data'][idx]:
            lab_seq[item] = 1

        for k in range(len(pred_seq)):
            if pred_seq[k] == 1 and lab_seq[k] == 1:
                TP_2 += 1
            if pred_seq[k] == 1 and lab_seq[k] == 0:
                FP_2 += 1
            if pred_seq[k] == 0 and lab_seq[k] == 1:
                FN_2 += 1

    if TP_1 == 0:
        p_1, r_1, f_1 = 0, 0, 0
    else:
        p_1, r_1, f_1 = 100*TP_1/(TP_1+FP_1), 100*TP_1 / \
            (TP_1+FN_1), 100*2*TP_1/(2*TP_1+FN_1+FP_1)

    if TP_2 == 0:
        p_2, r_2, f_2 = 0, 0, 0
    else:
        p_2, r_2, f_2 = 100*TP_2/(TP_2+FP_2), 100*TP_2 / \
            (TP_2+FN_2), 100*2*TP_2/(2*TP_2+FN_2+FP_2)

    return (p_1, r_1, f_1), (p_2, r_2, f_2)
