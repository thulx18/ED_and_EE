import os, random
import argparse

import numpy as np
import torch

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='DMBERT Code for NLP Event Detection Task')
        # file path
        parser.add_argument('--output_dir', default='./checkpoints',  help='the output dir for model checkpoints')
        parser.add_argument('--bert_dir', default='chinese-roberta-wwm-ext', help='bert dir for uer')
        parser.add_argument('--data_dir', default='./data', help='data dir for uer')
        parser.add_argument('--train_file', default='./train.json')
        parser.add_argument('--dev_file', default='./dev.json')
        # training
        parser.add_argument('--epochs', default=10, type=int, help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.1, type=float, help='drop out probability')
        parser.add_argument('--lr', default=2e-4, type=float, help='learning rate for the bert module')
        parser.add_argument('--bert_lr', default=2e-5, type=float,  help='learning rate for the module except bert')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--train_batch_size', default=4, type=int)
        # other
        parser.add_argument('--model_name', default='DMBERT', type=str)
        parser.add_argument('--num_class', default=66, type=int, help='number of event type')
        parser.add_argument('--gpu', type=int, default=0, help='gpu to use, -1 for cpu, other for gpu')
        parser.add_argument('--max_seq_len', default=256, type=int)
        parser.add_argument('--eval_batch_size', default=4, type=int)
        parser.add_argument('--swa_start', default=3, type=int, help='the epoch when swa start')
        parser.add_argument('--log_step', default=100, type=int, help='logging per step nums')
        parser.add_argument('--seed', default=123, type=int, help='random seed')

        args = parser.parse_args()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])
        # device
        self.device = torch.device(f'cuda:{self.gpu}' if self.gpu and torch.cuda.is_available() else "cpu")
        # file path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.train_path = os.path.join(self.data_dir, self.train_file)
        self.dev_path = os.path.join(self.data_dir, self.dev_file)
        # set random seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)  # set seed for cpu
        torch.cuda.manual_seed(self.seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(self.seed)  # set seed for all gpu


    def __str__(self):
        ret = ''
        for key in self.__dict__:
            ret = ret + f'{key} : {self.__dict__[key]}\n'
        return ret

if __name__ == '__main__':
    args = Config()
    print(args)