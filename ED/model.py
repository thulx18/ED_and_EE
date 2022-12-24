# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel


class DMBERT(nn.Module):
    def __init__(self, args):
        super(DMBERT, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        self.bert_config = self.bert.config
        self.out_dims = self.bert_config.hidden_size

        self.drop_out = nn.Dropout(args.dropout_prob)
        self.max_pooling = nn.MaxPool1d(args.max_seq_len)
        self.fc = nn.Linear(self.out_dims*2, args.num_class)

    def forward(self, data):
        token_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        maskL, maskR = data['maskL'], data['maskR']
        label = data['event_type'] if 'event_type' in data else None
        # bert
        outputs = self.bert(token_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # dynamic pooling
        conv = outputs[0].permute(2, 0, 1)  # bs * seq * hidden --> hidden * bs * seq
        L = (conv * maskL).transpose(0, 1)  # bs * hidden * seq
        R = (conv * maskR).transpose(0, 1)  # bs * hidden * seq
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.max_pooling(L).contiguous().view(token_ids.size(0), self.out_dims)   # bs * hidden
        pooledR = self.max_pooling(R).contiguous().view(token_ids.size(0), self.out_dims)   # bs * hidden
        pooled = torch.cat([pooledL, pooledR], 1)   # bs * (2 * hidden)
        pooled = pooled - torch.ones_like(pooled)
        # classify
        pooled = self.drop_out(pooled)
        logits = self.fc(pooled)    # bs * num_class

        return label, logits
