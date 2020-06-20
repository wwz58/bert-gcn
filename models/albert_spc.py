# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class ALBERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(ALBERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        hiddens, _ = self.bert(text_bert_indices, bert_segments_ids)
        hiddens = self.dropout(hiddens)
        hiddens = hiddens.max(1)[0]
        logits = self.dense(hiddens)
        return logits
