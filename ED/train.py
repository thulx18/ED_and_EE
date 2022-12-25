# -*- coding:utf-8 -*-
import os
import logging
import numpy as np

from config import Config
from data_helper import create_dataloader
from model import DMBERT
from utils import build_optimizer, evaluate

import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def train(args, model, train_loader, dev_loader):
    t_total = len(train_loader) * args.epochs
    best_f1 = 0.0

    loss_fn = CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(model, args, t_total)
    
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for _, batch in tqdm(enumerate(train_loader)):
            global_step += 1
            label, out = model(batch)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % args.log_step == 0:
                logging.info(f'[{args.model_name} train] epoch:{epoch}/{args.epochs} step:{global_step}/{t_total} loss:{loss.item():.6f}')
        # evaluate on dev
        total_loss, f1, acc = evaluate(model, dev_loader)
        print(f'[{args.model_name} dev] loss:{total_loss:.6f} accuracy:{acc:.4f} f1:{f1:.4f}')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, os.path.join(args.output_dir, f'{args.model_name}.model'))
    
    logging.info(f'######## {args.model_name} 完成训练 ########')
    

if __name__ == '__main__':
    args = Config()
    train_loader = create_dataloader(args, args.train_path, 'train')
    dev_loader = create_dataloader(args, args.dev_path, 'eval')
    model = DMBERT(args)

    if args.gpu >= 0:
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    logging.info(f'######## {args.model_name} 开始训练 ########')
    train(args, model, train_loader, dev_loader)
