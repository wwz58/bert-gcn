import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig

from layers.attention import Attention, NoQueryAttention
from layers.squeeze_embedding import SqueezeEmbedding


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, step=2, drop=0.0):
        super(GNN, self).__init__()
        self.step = step
        self.drop = drop
        self.W1 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        if input_dim == hidden_dim:
            self.W2 = self.W1
        else:
            self.W2 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.Tensor(hidden_dim))
        if input_dim == hidden_dim:
            self.b2 = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.b2 = self.b1

    def forward(self, X, A):
        """B L H, B L L -> B H L"""
        for i in range(self.step):
            if i == 0:
                X = torch.matmul(X, self.W1)
            else:
                X = torch.matmul(X, self.W2)
            normed_A = A / A.sum(1, keepdim=True)[0]
            X = torch.matmul(A, X)
            if i == 0:
                X = X + self.b1
            else:
                X = X + self.b2
            if i < self.step - 1:
                X = F.relu(X)
                if self.drop > 0:
                    X = F.dropout(X)
        return X


class GNN_RELU(nn.Module):
    def __init__(self, input_dim, hidden_dim, step=2, drop=0.0):
        super(GNN_RELU, self).__init__()
        self.step = step
        self.drop = drop
        self.W1 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        if input_dim == hidden_dim:
            self.W2 = self.W1
        else:
            self.W2 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.Tensor(hidden_dim))
        if input_dim == hidden_dim:
            self.b2 = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.b2 = self.b1

    def forward(self, X, A):
        """B L H, B L L -> B H L"""
        normed_A = A / (A.sum(1, keepdim=True)[0])
        for i in range(self.step):
            if i == 0:
                X = torch.matmul(X, self.W1)
            else:
                X = torch.matmul(X, self.W2)

            X = torch.matmul(normed_A, X)
            if i == 0:
                X = X + self.b1
            else:
                X = X + self.b2
            X = F.relu(X)
            if self.drop > 0:
                X = F.dropout(X)
        return X


class GNN_RELU_DIFF(nn.Module):
    def __init__(self, input_dim, hidden_dim, step=2, drop=0.0):
        super(GNN_RELU_DIFF, self).__init__()
        self.step = step
        self.drop = drop
        self.Ws = nn.ParameterList(
            [nn.Parameter(torch.Tensor(input_dim, hidden_dim))])
        self.bs = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_dim))])
        for i in range(1, step):
            self.Ws.append(nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)))
            self.bs.append(nn.Parameter(torch.Tensor(hidden_dim)))

    def forward(self, X, A):
        """B L H, B L L -> B H L"""
        normed_A = A / (A.sum(1, keepdim=True)[0])
        for i in range(self.step):
            X = torch.matmul(X, self.Ws[i])
            X = torch.matmul(normed_A, X)
            X = X + self.bs[i]
            X = F.relu(X)
            if self.drop > 0:
                X = F.dropout(X)
        return X


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        # zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len),
        #                                     dtype=np.float32), dtype=torch.float32).to(self.opt.device)
        zero_tensor = 0
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BERT_ALBERT_GCN(nn.Module):
    """
    bert:做encoder 输入：context_asp_ids, context_asp_seg_ids, context_asp_att_mask, adj_matrix, start, end, polarity
    glove+bom做encoder 输入同上 init 时需要embedding_matrix_pkl(conf.tokenizer_pkl),需要embedding层
    除encoder层外一样，能够控制是否加入 pos+pct gnn+pct short_cut
                    /-------short cut +-------\  /---no sa---\
    squeeze -> bert -> norm + gcn -> pct + gelu -> sa + tanh -> bert_pooler -> fc -> logits
    """

    def __init__(self, bert_or_embedding_matrix, opt):
        super(BERT_ALBERT_GCN, self).__init__()
        self.opt = opt
        bert = bert_or_embedding_matrix
        self.context_bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        if not opt.no_gnn:
            self.gcn = GNN_RELU_DIFF(opt.bert_dim, opt.bert_dim,
                                     step=opt.gnn_step, drop=0.0)
            self.encoder_fn = nn.Linear(opt.bert_dim, opt.bert_dim)
            self.norm = LayerNorm(opt.bert_dim)
        if not opt.no_sa:
            self.bert_self_att = SelfAttention(bert.config, opt)
        if self.opt.pool_tp == 'bert_pool':
            self.bert_pooler = BertPooler(bert.config)
        elif self.opt.pool_tp == 'max_pool':
            self.pool_fc = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.fc = nn.Linear(opt.bert_dim, 3)

    def forward(self, inputs):
        # 解开inputs:context=text+asp
        context, context_seg = inputs[0], inputs[1]
        asp_start, asp_end, adj_matrixs = inputs[2], inputs[3], inputs[4]
        attention_mask = inputs[5]
        for e in inputs:
            assert not torch.isnan(e).any()
        # 获得context_len ,并进行squeezeEmbedding
        context_len = torch.sum(context != 0, dim=-1).long()
        context = self.squeeze_embedding(context, context_len)
        context_ids = context
        context_seg = self.squeeze_embedding(context_seg, context_len)
        attention_mask = self.squeeze_embedding(attention_mask, context_len)
        # context 进行 bert/glove encode
        context, _ = self.context_bert(
            context, context_seg, attention_mask)

        if not self.opt.no_gnn:
            batch_size, seq_len = context.size()[:-1]
            # +short_cut(
            highway = context
            context = self.norm(context)

            # +gnn【cls sep连所有，context按照语法树连成无向图，asp连所有】
            A = adj_matrixs[:, :seq_len, :seq_len]
            context = self.gcn(context, A)

            # + pct + gelu
            context = self.encoder_fn(context)
            context = F.gelu(context)

            # +short_cut)
            if not self.opt.no_short_cut:
                context += highway

        # + sa
        if not self.opt.no_sa:
            rep = self.bert_self_att(context)
        else:
            rep = context
        # + bert_pooler
        if self.opt.pool_tp == 'bert_pool':
            rep = self.bert_pooler(rep)
        elif self.opt.pool_tp == 'max_pool':
            # B L
            mask = torch.zeros(rep.size()[:-1]).to(self.opt.device)
            for i, (seq, s, e, true_len) in enumerate(zip(context_ids, asp_start, asp_end, context_len)):
                seq = seq.tolist()
                first_sep_idx = seq.index(102)
                mask[i, 0] = 1
                mask[i, s:e] = 1
                mask[i, first_sep_idx + 1:true_len] = 1
            rep = rep * mask.unsqueeze(-1)  # B L H * B L 1-->B L H
            rep = rep.max(1)[0]  # B H
            rep = self.pool_fc(rep)
            rep = F.tanh(rep)
        # fc
        out = self.fc(rep)
        return out
