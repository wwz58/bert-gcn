# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import copy
import random
import numpy
from collections import defaultdict

from pytorch_transformers import BertModel
from transformers import AlbertModel

import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.data.dataloader import default_collate

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Albert, Tokenizer4Bert, ABSADataset,Tokenizer4AlbertGcn,Tokenizer4BertGcn,ABSAGcnData

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT, ALBERT_SPC, BERT_ALBERT_GCN, BERT_GCN_V1
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from models import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if 'v1' in opt.model_name and 'albert' in opt.model_name:
            tokenizer = Tokenizer4AlbertGcn(opt.max_seq_len, opt.pretrained_bert_name)
            bert = None
            self.model = opt.model_class(bert, opt).to(opt.device)
        elif 'v1' in opt.model_name and 'bert' in opt.model_name:
            tokenizer = Tokenizer4BertGcn(opt.max_seq_len, opt.pretrained_bert_name)
            bert = None
            self.model = opt.model_class(bert, opt).to(opt.device)
        elif 'albert_gcn' in opt.model_name:
            tokenizer = Tokenizer4AlbertGcn(opt.max_seq_len, opt.pretrained_bert_name)
            bert = AlbertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        elif 'bert_gcn' in opt.model_name:
            tokenizer = Tokenizer4BertGcn(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        elif 'albert' in opt.model_name:
            tokenizer = Tokenizer4Albert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = AlbertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        elif 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        if 'bert' in opt.model_name and opt.freeze_bert:
            try:
                self.model.bert.requires_grad = False
            except:
                self.model.context_bert.requires_grad = False

        if 'gcn' in opt.model_name:
            self.trainset = ABSAGcnData(opt.dataset_file['train'], tokenizer, debug=opt.debug, from_xml=opt.from_xml)
            self.testset = ABSAGcnData(opt.dataset_file['test'], tokenizer, debug=opt.debug, from_xml=opt.from_xml)
        else:
            self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer, debug=opt.debug)
            self.testset = ABSADataset(opt.dataset_file['test'], tokenizer, debug=opt.debug)
        assert 0 <= opt.valset_ration < 1
        if opt.valset_ration > 0:
            valset_len = int(len(self.trainset) * opt.valset_ration)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        val_f1_at_best_val_acc = 0
        best_val_acc_epoch = 0
        tr_acc_at_best_epoch = 0
        tr_f1_at_best_epoch = 0

        global_step = 0
        last_path = None
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                tr_acc_at_best_epoch, tr_f1_at_best_epoch = self._evaluate_acc_f1(train_data_loader)
                best_val_acc_epoch = epoch
                val_f1_at_best_val_acc = val_f1

                if last_path is not None:
                    os.system('rm -rf {}'.format(last_path))
                    logging.info('>> succefully rm last best path: {}'.format(last_path))
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                last_path = path
                logger.info('>> saved new best model pkl in: {}'.format(path))
            # if val_f1 > max_val_f1:
            #     max_val_f1 = val_f1
        stat = {'max_val_acc' : max_val_acc,
            'val_f1_at_best_val_acc' : val_f1_at_best_val_acc,
            'best_val_acc_epoch' : best_val_acc_epoch,
            'tr_acc_at_best_epoch' : tr_acc_at_best_epoch,
            'tr_f1_at_best_epoch' : tr_f1_at_best_epoch}
        return path, stat

    def _train_eval_step(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        val_f1_at_best_val_acc = 0
        best_val_acc_epoch = 0
        tr_acc_at_best_epoch = 0
        tr_f1_at_best_epoch = 0

        global_step = 0
        last_path = None
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                    val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
                    logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        tr_acc_at_best_epoch, tr_f1_at_best_epoch = self._evaluate_acc_f1(train_data_loader)
                        best_val_acc_epoch = epoch
                        val_f1_at_best_val_acc = val_f1

                        if last_path is not None:
                            os.system('rm -rf {}'.format(last_path))
                            logging.info('>> succefully rm last best path: {}'.format(last_path))
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                        torch.save(self.model.state_dict(), path)
                        last_path = path
                        logger.info('>> saved new best model pkl in: {}'.format(path))
            # if val_f1 > max_val_f1:
            #     max_val_f1 = val_f1
        stat = {'max_val_acc' : max_val_acc,
            'val_f1_at_best_val_acc' : val_f1_at_best_val_acc,
            'best_val_acc_epoch' : best_val_acc_epoch,
            'tr_acc_at_best_epoch' : tr_acc_at_best_epoch,
            'tr_f1_at_best_epoch' : tr_f1_at_best_epoch}
        return path, stat

    def _evaluate_acc_f1(self, data_loader, save_res_path=None):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        texts, asps = [], []
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                texts += t_sample_batched['text']
                asps += t_sample_batched['asp_text']
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        y_true, y_pred = t_targets_all.cpu().tolist(), torch.argmax(t_outputs_all, -1).cpu().tolist()
        f1 = metrics.f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')

        if save_res_path is not None:
            with open(save_res_path, 'w') as f:
                for line, asp, y, pred in zip(texts, asps, y_true, y_pred):
                    f.write(line+'\n')
                    f.write(asp+'\n')
                    f.write(','.join([str(y), str(pred)])+'\n')

        return acc, f1

    @staticmethod
    def my_collate_fn(data):
        string_keys = set(['text', 'asp_text'])
        string_fields = defaultdict(list)
        data_copy = copy.deepcopy(data)
        for k in string_keys:
            for sample in data_copy:
                string_fields[k].append(sample.pop(k, None))
        batch = default_collate(data_copy)
        batch.update(string_fields)
        return batch
        
    def run(self, trainset=None, valset=None, testset=None):
        # run one model - one train - one test then record stat and param in one csv
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if trainset is None:
            trainset, valset, testset = self.trainset, self.valset, self.testset
        train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True, collate_fn=Instructor.my_collate_fn)
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.batch_size, shuffle=False, collate_fn=Instructor.my_collate_fn)
        val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size, shuffle=False, collate_fn=Instructor.my_collate_fn)
        
        self._reset_params()

        if self.opt.step_eval:
            best_model_path, stat = self._train_eval_step(criterion, optimizer, train_data_loader, val_data_loader)
        else:
            best_model_path, stat = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        
        time_stamp = strftime("%y%m%d-%H%M%S")
        stat['time_stamp'] = time_stamp
        csv_path = 'log/{}.csv'.format(time_stamp)
        test_res_path = 'test_res/{}.txt'.format(time_stamp)

        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader, save_res_path=test_res_path)
        stat['test_acc'] = test_acc
        stat['test_f1'] = test_f1
        stat['tr_data_len'] = len(trainset)
        stat['val_data_len'] = len(valset)
        stat['te_data_len'] = len(testset)
        stat.update(vars(self.opt))

        pd.DataFrame([stat]).to_csv(csv_path, index=False)
        os.system('rm -rf {}'.format(best_model_path))
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

    def cv_run(self):
        # valset_ration = 0 must!
        valset_len = len(self.trainset) // self.opt.cross_val_fold
        splitedsets = random_split(self.trainset, tuple([valset_len] * (self.opt.cross_val_fold - 1) + [len(self.trainset) - valset_len * (self.opt.cross_val_fold - 1)]))

        # all_test_acc, all_test_f1 = [], []
        for fid in range(self.opt.cross_val_fold):
            logger.info('fold : {}'.format(fid))
            logger.info('>' * 100)
            trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
            valset = splitedsets[fid]
            self.run(trainset, valset, self.testset)

    def random_run(self, n_repeat):
        for i in range(n_repeat):
            logger.info('random runs : {}'.format(i))
            logger.info('>' * 100)
            self.run()

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_gcn', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=5, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='/mnt/2/ckh/BERT系列预训练模型/bert-base-uncased/', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=2, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ration', default=0, type=float, help='set ration between 0 and 1 for validation support')
    parser.add_argument('--cross_val_fold', default=0, type=int, help='k-fold cross validation')
    parser.add_argument('--n_repeat', default=0, type=int, help='num of times random run')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--from_xml', action='store_true', help='from_xml')
    parser.add_argument('--step_eval', action='store_true', help='step_eval')
    parser.add_argument('--freeze_bert', default=False, type=bool, help='freeze_bert')

    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    # The following parameters are only valid for the bert_gcn model
    parser.add_argument('--no_gnn', default=False, type=bool, help='no_gnn')
    parser.add_argument('--no_sa', default=False, type=bool, help='no_sa')
    parser.add_argument('--pool_tp', default='bert_pool', type=str, help='pool_to to get rep, bert_pool or max_pool')
    parser.add_argument('--gnn_step', default=2, type=int, help='gnn_step')
    parser.add_argument('--no_short_cut', default=False, type=bool, help='no_short_cut')

    opt = parser.parse_args()

    # opt.model_name = 'bert_gcn'
    # opt.num_epoch = 2
    # opt.valset_ration=0.2
    # opt.from_xml = True
    # opt.debug = True

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_gcn_v2': BERT_GCN_V2,
        'bert_gcn_v1': BERT_GCN_V1,
        'albert_gcn_v1': BERT_GCN_V1,
        'bert_gcn': BERT_ALBERT_GCN,
        'albert_gcn': BERT_ALBERT_GCN,
        'albert_spc': ALBERT_SPC,
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    ori_dataset_files = {
        'twitter': {
            'train': '../bert_gcn/datasets/acl-14-short-data/train.raw',
            'test': '../bert_gcn/datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': '../bert_gcn/datasets/semeval14/Restaurants_Train.xml',
            'test': '../bert_gcn/datasets/semeval14/Restaurants_Test_Gold.xml'
        },
        'laptop': {
            'train': '../bert_gcn/datasets/semeval14/Laptops_Train.xml',
            'test': '../bert_gcn/datasets/semeval14/Laptops_Test_Gold.xml'
        }

    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    
    input_colses = {
        'bert_gcn_v2': ['text_bert_indices', 'bert_segments_ids','asp_start','asp_end','context_asp_adj_matrix','context_asp_attention_mask'],
        'albert_gcn_v1': ['text_bert_indices', 'bert_segments_ids','asp_start','asp_end','context_asp_adj_matrix','context_asp_attention_mask'],
        'bert_gcn_v1': ['text_bert_indices', 'bert_segments_ids','asp_start','asp_end','context_asp_adj_matrix','context_asp_attention_mask'],
        'bert_gcn': ['text_bert_indices', 'bert_segments_ids','asp_start','asp_end','context_asp_adj_matrix','context_asp_attention_mask'],
        'albert_gcn': ['text_bert_indices', 'bert_segments_ids','asp_start','asp_end','context_asp_adj_matrix','context_asp_attention_mask'],
        'albert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset] if not opt.from_xml else ori_dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    if opt.cross_val_fold > 0 and opt.valset_ration > 0:
        ins.cv_run()
    elif opt.n_repeat > 0 and opt.valset_ration <= 0:
        ins.random_run(opt.n_repeat)
    else:
        ins.run()


if __name__ == '__main__':
    main()
