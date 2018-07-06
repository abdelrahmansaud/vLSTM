# Implementation of Variational LSTM from paper "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks"
# https://arxiv.org/pdf/1512.05287

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class vLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, p_w=0.0, p_u=0.0):
        super(pLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.p_w = p_w
        self.p_u = p_u
        self.masked = False
        self.dropout = (0 < p_w < 1) or (0 < p_u < 1)

        if self.dropout:
            self.wi_dropout = nn.Dropout(p_w)
            self.wi_mask = torch.ones(input_size).cuda()
            self.wf_dropout = nn.Dropout(p_w)
            self.wf_mask = torch.ones(input_size).cuda()
            self.wo_dropout = nn.Dropout(p_w)
            self.wo_mask = torch.ones(input_size).cuda()
            self.wg_dropout = nn.Dropout(p_w)
            self.wg_mask = torch.ones(input_size).cuda()

            self.ui_dropout = nn.Dropout(p_u)
            self.ui_mask = torch.ones(hidden_size).cuda()
            self.uf_dropout = nn.Dropout(p_u)
            self.uf_mask = torch.ones(hidden_size).cuda()
            self.uo_dropout = nn.Dropout(p_u)
            self.uo_mask = torch.ones(hidden_size).cuda()
            self.ug_dropout = nn.Dropout(p_u)
            self.ug_mask = torch.ones(hidden_size).cuda()

        
        self.i_w = nn.Linear(input_size, hidden_size)
        self.i_u = nn.Linear(hidden_size, hidden_size)
        
        self.f_w = nn.Linear(input_size, hidden_size)
        self.f_u = nn.Linear(hidden_size, hidden_size)
        
        self.o_w = nn.Linear(input_size, hidden_size)
        self.o_u = nn.Linear(hidden_size, hidden_size)
        
        self.g_w = nn.Linear(input_size, hidden_size)
        self.g_u = nn.Linear(hidden_size, hidden_size)
        
        self.h_0 = torch.zeros((1, self.hidden_size)).cuda()
        self.c_0 = torch.zeros((1, self.hidden_size)).cuda()

        
    def forward(self, inp, hiddens=None):
        if hiddens is None:
            hiddens = self.init_hidden()
        self.h_0 = hiddens[0]
        self.c_0 = hiddens[1]
        
        if self.dropout and (not self.masked):
            self.wi_mask = self.wi_dropout(self.wi_mask)
            self.wf_mask = self.wf_dropout(self.wf_mask)
            self.wo_mask = self.wo_dropout(self.wo_mask)
            self.wg_mask = self.wg_dropout(self.wg_mask)

            self.ui_mask = self.ui_dropout(self.ui_mask)
            self.uf_mask = self.uf_dropout(self.uf_mask)
            self.uo_mask = self.uo_dropout(self.uo_mask)
            self.ug_mask = self.ug_dropout(self.ug_mask)
            self.masked = True

        if self.dropout and self.training:
            i = F.sigmoid(self.i_u(self.h_0.mul(self.ui_mask)) + self.i_w(inp.mul(self.wi_mask)))
            f = F.sigmoid(self.f_u(self.h_0.mul(self.uf_mask)) + self.f_w(inp.mul(self.wf_mask)))
            o = F.sigmoid(self.o_u(self.h_0.mul(self.uo_mask)) + self.o_w(inp.mul(self.wo_mask)))
            g = F.tanh(self.g_u(self.h_0.mul(self.ug_mask)) + self.g_w(inp.mul(self.wg_mask)))
            c = f.mul(self.c_0) + i.mul(g)
            h = o.mul(F.tanh(c))
                
        else:
            i = F.sigmoid(self.i_u(self.h_0) + self.i_w(inp))
            f = F.sigmoid(self.f_u(self.h_0) + self.f_w(inp))
            o = F.sigmoid(self.o_u(self.h_0) + self.o_w(inp))
            g = F.tanh(self.g_u(self.h_0) + self.g_w(inp))
            c = f.mul(self.c_0) + i.mul(g)
            h = o.mul(F.tanh(c))
        
        return h, c
    
    def init_hidden(self):
        self.h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        self.c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()

        return h_0, c_0

