# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import re
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from transformers import AlbertTokenizer
import spacy
from spacy.tokens import Doc
from xml.etree import ElementTree as ET


label2id = {t: i for i, t in enumerate(['negative', 'neutral', 'positive'])}

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Albert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = AlbertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer, debug=False, from_xml=None):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            if debug and i>120:
                break
            text = lines[i].strip()
            asp_text = lines[i + 1].strip()
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text': text,
                'asp_text': asp_text,
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


replace_p = re.compile(r'\xa0')
add_blank_p = re.compile(r"([][!\"#$%&\\'()*+,\-./:;<=>?@])")
blank_p = re.compile(r'\s+')

def clean_s(s):
    s = replace_p.sub('', s)
    s = add_blank_p.sub(r' \1 ', s)
    s = blank_p.sub(' ', s)
    return s.strip()

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

class Tokenizer4BertGcn:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

class Tokenizer4AlbertGcn:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = AlbertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

def read_seg_ex_with_clean(fn, label2id=None, debug=False):
    with open(fn) as fin:
        lines = fin.readlines()
    for i in range(0, len(lines), 3):
        if debug and i > 90:
            return
            yield
        text = lines[i]
        asp_text = lines[i + 1]
        text_left, _, text_right = [s.lower().strip()
                                    for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        polarity = lines[i + 2].strip()
        polarity = int(polarity)+1
        text_left, aspect, text_right = [clean_s(s) for s in [text_left, aspect, text_right]]
        yield [i, text, asp_text, text_left, aspect, text_right, polarity]

def read_ex_with_clean(fn, label2id, debug=False):
    tree = ET.parse(fn)
    root = tree.getroot()
    replace_p = re.compile(r'\xa0')
    add_blank_p = re.compile(r"([][!\"#$%&\\'()*+,\-./:;<=>?@])")
    blank_p = re.compile(r'\s+')
    for i, sentence in enumerate(root.findall('sentence')):
        if debug and i > 90:
            return 
            yield
        id = sentence.get('id')
        text = sentence.find('text').text
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is not None:
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                term = aspectTerm.get('term')
                asp_text = term
                polarity = aspectTerm.get('polarity')
                if polarity == 'conflict':
                    continue
                polarity = label2id[polarity]
                start = int(aspectTerm.get('from'))
                end = int(aspectTerm.get('to'))
                left = text[:start]
                right = text[end:]
                left, term, right = [replace_p.sub('', s) for s in [
                    left, term, right]]
                left, term, right = [add_blank_p.sub(r' \1 ', s) for s in [
                    left, term, right]]
                left, term, right = [blank_p.sub(' ', s) for s in [
                    left, term, right]]
                example_text = (left + ' $T$ ' + right).strip()
                yield [id, example_text, asp_text, left.strip(), term.strip(), right.strip(), polarity]

class ABSAGcnData(Dataset):
    def __init__(self, fn, tokenizer, debug=False, from_xml=False):

        self.nlp = spacy.load('en_core_web_lg', disable=['tagger', 'ner'])
        self.nlp.tokenizer = WhitespaceTokenizer(self.nlp.vocab)
        self.data = []
        # with open(fn) as fin:
        #     lines = fin.readlines()
        # for i in range(0, len(lines), 3):
        #     if debug and i > 120:
        #         break
        #     text = lines[i]
        #     asp_text = lines[i + 1]
        #     left, _, right = [s.lower().strip()
        #                                 for s in lines[i].partition("$T$")]
        #     term = lines[i + 1].strip()
        #     left, term , right = [clean_s(s) for s in [left, term, right]]

        #     polarity = lines[i + 2].strip()
        #     polarity = int(polarity) + 1
        read_ex = None
        if from_xml:
            read_ex = read_ex_with_clean
        else:
            read_ex = read_seg_ex_with_clean
        for _, text, asp_text, left, term, right, polarity in read_ex(fn, label2id=label2id, debug=debug):
                
            # 3部分用空格连起来 使用white space tokenizer 和 spacy 获得 ori_adj [ ori_len ori_len]
            spacy_s = ' '.join([sent for sent in [left, term, right] if sent])
            ori_len = len(spacy_s.split())
            ori_adj = np.eye(ori_len).astype('float32')
            for t in self.nlp(spacy_s):
                for child in t.children:
                    ori_adj[t.i][child.i] = 1
                    ori_adj[child.i][t.i] = 1
            # 3部分分别 bert tokenize 按照space 分的每个词 tokenize 得到 asp_start, asp_end, bert_tokens 和 tok2ori_map
            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left.split()):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)
                    left_tok2ori_map.append(ori_i)
            asp_start = len(left_tokens)
            offset = len(left.split())
            for ori_i, w in enumerate(term.split()):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term.split())
            for ori_i, w in enumerate(right.split()):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    # right_tok2ori_map.append(ori_i)
                    right_tok2ori_map.append(ori_i+offset)

            # truncate bert_tokens 和 tok2ori_map
            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
            # 构造 adj:[truncate_tok_len, truncate_tok_len] 的 tok_adj
            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]

            # 调用tokenizer对将token转换为id，对left right进行 padding 到max_len，
            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [
                0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            context_asp_attention_mask = [
                1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            # 生成相应的seg_id 和 att_mask
            context_asp_seg_ids = np.asarray(
                context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(
                context_asp_attention_mask, dtype='int64')
            # pad adj
            context_asp_adj_matrix = np.ones(
                (tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
            pad_adj = np.ones(
                (context_asp_len, context_asp_len)).astype('float32')
            pad_adj[1:context_len + 1, 1:context_len + 1] = tok_adj
            context_asp_adj_matrix[:context_asp_len,
                                   :context_asp_len] = pad_adj
            data = {
                'text': text,
                'asp_text': asp_text,
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'context_asp_attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'context_asp_adj_matrix': context_asp_adj_matrix,
                'polarity': polarity,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
