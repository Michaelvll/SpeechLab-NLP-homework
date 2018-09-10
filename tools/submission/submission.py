#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:25:16 2018

@author: wang
"""
import os

import utils.data_reader as data_reader
import utils.acc as acc

#data path
prex_path = os.getcwd()
data_root = os.path.join(prex_path, 'data')
result_root = os.path.join(prex_path, 'result')

#prepare the data
slot_tag_vocab_dir = os.path.join(data_root, 'lab')
intent_tag_vocab_dir = os.path.join(data_root, 'lab_unali')
test_data_dir = os.path.join(data_root, 'test')

slot_tag_to_idx, idx_to_slot_tag = data_reader.read_vocab_file(slot_tag_vocab_dir, no_pad=False, no_unk=False)
intent_tag_to_idx, idx_to_intent_tag = data_reader.read_vocab_file(intent_tag_vocab_dir, no_pad=True, no_unk=True)
test_slot_tags, test_intent_tags = data_reader.read_seqtag_data(test_data_dir, slot_tag_to_idx, intent_tag_to_idx)

#generating intent submission
out_intent_path = os.path.join(result_root, 'submission_intent.csv')
line_id = 1
with open(out_intent_path,'w') as f:
    f.write('Id,Expected\n')
    for line in test_intent_tags['data']:
        for i in range(len(intent_tag_to_idx)):
            if i in line:
                f.write(str(line_id)+',1\n')
            else:
                f.write(str(line_id)+',0\n')
            line_id += 1

#prepare the type list
#there are 83 'B' slot tags, 44 'I' slot tags, 1 'O' tag
#44 slots have max length of 1, 39 slots have max length of 3
B_type_list, I_type_list = [],[]
for item in slot_tag_to_idx:
    if item[0] == 'B':
        B_type_list.append(item[2:])
    if item[0] == 'I':
        I_type_list.append(item[2:])
type_list = []
for item in B_type_list:
    if item in I_type_list:
        type_list.append((item,3))
    else:
        type_list.append((item,1))

out_slot_path = os.path.join(result_root, 'submission_slot.csv')
line_id = 1  
with open(out_slot_path,'w') as f:
    f.write('Id,Expected\n')
    for line in test_slot_tags['data']:
        all_chunks = []
        sentence_length = len(line)
        for Type,slot_length in type_list:
            for i in range(1,sentence_length+1):
                for j in range(i,max(i+slot_length,sentence_length+1)):
                    all_chunks.append((i,j,Type))
        lab_seq = [idx_to_slot_tag[slot] for slot in line]
        label_chunks = acc.get_chunks(['O']+lab_seq+['O'])
        for k in range(len(all_chunks)):
            if all_chunks[k] in label_chunks:
                f.write(str(line_id)+',1\n')
            else:
                f.write(str(line_id)+',0\n')
            line_id += 1
  