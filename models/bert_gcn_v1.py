# from layers.gcn import GCN
# from layers.layernorm import LayerNorm
from torch import nn
import torch.nn.functional as F
import torch
from pytorch_transformers.modeling_bert import BertSelfAttention
from transformers import AutoModel, AutoConfig

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
        super().__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        # zero_tensor is att_mask should be broadestable to B L L H all 0 means no mask out
        zero_tensor = 0
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class BERT_GCN_V1(nn.Module):
    """
        /   no_short_cut \
            /no_gnn\         /  no_sa \
    bert -> gcn -> + -> LN -> SA -> tanh -> extract cls rep -> fc
         \________/                      \__ extract entity rep -> max_pool -> fc

    1.  no_gnn ==> no_short_cut
    2.  has gnn ==> has sa
    3.  choices:
            普通bert no_gcn + no_sa + bert_pool
            测试gcn 的short cut：has_gcn + no_short_cut + has_sa
            完整has_gcn + has_short_cut + has_sa
    """

    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        if opt.debug:
            conf = AutoConfig.from_pretrained(opt.pretrained_bert_name)
            self.bert = AutoModel.from_config(conf)
        else:
            self.bert = AutoModel.from_pretrained(
                opt.pretrained_bert_name)
        if not opt.no_gnn:
            self.gcn = GNN_RELU_DIFF(opt.bert_dim, opt.bert_dim,
                           step=opt.gnn_step, drop=opt.dropout)
            if not opt.no_short_cut:
                self.layernorm = LayerNorm(opt.bert_dim)
        if not opt.no_sa:
            self.sa = SelfAttention(self.bert.config, opt)
        self.drop = nn.Dropout(opt.dropout)
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        inp_ids, seg_ids, asp_s, asp_e, adj, att_mask = inputs
        emb, pool_out = self.bert(inp_ids, seg_ids, att_mask)
        if not self.opt.no_gnn:
            if not self.opt.no_short_cut:
                highway = emb
            emb = self.gcn(emb, adj)
            if not self.opt.no_short_cut:
                emb = emb + highway
                emb = self.layernorm(emb)
        if not self.opt.no_sa:
            emb = self.sa(emb)
        if self.opt.pool_tp == 'cls':
            rep = emb[:, 0, :]
        elif self.opt.pool_tp == 'max_pool':
            B, L, H = emb.size()
            mask = torch.zeros(B, L)
            for i, (s, e) in enumerate(zip(asp_s, asp_e)):
                mask[i, s:e] = 1.0
            mask = mask.unsqueeze(-1)
            rep = emb * mask
            rep = rep.max(dim=1)[0]
        elif self.opt.pool_tp == 'bert_pool':
            rep = pool_out
        else:
            raise NotImplementedError
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits
