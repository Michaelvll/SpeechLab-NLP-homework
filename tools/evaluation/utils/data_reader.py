#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:17:30 2018

@author: wang
"""

import torch
import operator
import json


def read_vocab_file(vocab_path, no_pad=False, no_unk=False, separator=':'):
    '''file format:"word : idx" '''
    word2idx, idx2word = {}, {}
    if not no_pad:
        word2idx['<pad>'] = len(word2idx)
        idx2word[len(idx2word)] = '<pad>'
    if not no_unk:
        word2idx['<unk>'] = len(word2idx)
        idx2word[len(idx2word)] = '<unk>'
    with open(vocab_path, 'r') as f:
        for line in f:
            if separator in line:
                word, idx = line.strip().split(' '+separator+' ')
                idx = int(idx)
            else:
                word = line.strip()
                idx = len(word2idx)
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
    return word2idx, idx2word


def read_seqtag_data(data_path, slot_tag2idx, intent_tag2idx, separator=':'):
    '''Read data from file'''
    slot_tag_seqs = []
    intent_tag_seqs = []

    with open(data_path, 'r') as f:
        for line in f:
            slot_tag_line, intent = line.strip('\n\r').split(' <=> ')
            if slot_tag_line == "":
                continue
            slot_tag_seq, intent_tag_seq = [], []
            for item in slot_tag_line.split(' '):
                word, tag = item.split(':')
                slot_tag_seq.append(
                    slot_tag2idx[tag] if tag in slot_tag2idx else slot_tag2idx['<unk>'])
            slot_tag_seqs.append(slot_tag_seq)
            for item in intent.split(';'):
                if item == '':
                    break
                intent_tag_seq.append(intent_tag2idx[item])
            intent_tag_seqs.append(intent_tag_seq)

    slot_tag_labels = {'data': slot_tag_seqs}
    intent_tag_labels = {'data': intent_tag_seqs}

    return slot_tag_labels, intent_tag_labels
