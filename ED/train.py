# -*- coding:utf-8 -*-
from imp import load_module
import logging
from config import config
from torch.utils.data import DataLoader
from dataset import dataset
from process.processor import Processor
from models.DMBERT import DMBERT
from models.CBiLSTM import CBiLSTM
from models.DMCNN import DMCNN
from torch.nn import CrossEntropyLoss
import torch
import transformers
import sklearn
import numpy as np
from tqdm import tqdm
from utils import commonUtils
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import set_seed

args = config.Args().get_parser()
logger = logging.getLogger(__name__)

get_class = lambda attr, name: getattr(__import__("{}.{}".format(attr, name), fromlist=["dummy"]), name)


def train(args, train_loader, dev_loader, model):
    device = torch.device("cuda:{}".format(args.gpu_ids) if args.gpu_ids and torch.cuda.is_available() else "cpu")
    model.to(device)

    model_name = model.__class__.__name__

    commonUtils.set_logger(os.path.join(args.log_dir, model_name+'.log'))

    print('######## {} 开始训练 ########'.format(model_name))
    # 开始训练
    t_total = len(train_loader) * args.train_epochs  # 总的执行次数
    best_f1 = 0.0

    loss_fct = CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=int(args.warmup_proportion * t_total),
                                                             num_training_steps=t_total)

    global_step = 0
    for epoch in range(args.train_epochs):
        print('第{}个epoch开始'.format(epoch))
        for step, batch_data in tqdm(enumerate(train_loader)):
            model.train()
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            global_step += 1
            label, out = model(**batch_data)
            loss = loss_fct(out, label)

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            # print('[{} train] epoch:{}/{} step:{}/{} loss:{:.6f}'.format(model_name, epoch, args.train_epochs,
            #                                                           global_step, t_total, loss.item()))
            if global_step % args.log_step == 0:
                logger.info('[{} train] epoch:{}/{} step:{}/{} loss:{:.6f}'.format(model_name, epoch, args.train_epochs,
                                                                            global_step, t_total, loss.item()))
        # 进行验证
        total_loss, f1, acc = dev(args, model, dev_loader)
        print('[{} dev] loss:{:.6f} accuracy:{:.4f} f1:{:.4f}'.format(
            model_name, total_loss, acc, f1))
        logger.info('[{} dev] loss:{:.6f} accuracy:{:.4f} f1:{:.4f}'.format(
            model_name, total_loss, acc, f1))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, "./checkpoints/{}.model".format(model_name))
    print('######## {} 完成训练 ########'.format(model_name))


def dev(args, model, dev_loader):
    device = torch.device("cuda:{}".format(args.gpu_ids) if args.gpu_ids and torch.cuda.is_available() else "cpu")
    model.to(device)

    model_name = model.__class__.__name__
    y_true, y_pre = None, None
    total_loss = 0
    model.eval()
    loss_fct = CrossEntropyLoss()  # 损失函数
    with torch.no_grad():
        for eval_step, batch_data in tqdm(enumerate(dev_loader)):
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            label, out = model(**batch_data)
            loss = loss_fct(out, label)
            total_loss += loss.item()   # 计算总的损失
            out_np = np.argmax(out.detach().cpu().numpy(), axis=1)
            label_np = np.argmax(label.detach().cpu().numpy(), axis=1)
            if y_pre is None:
                y_true = label_np
                y_pre = out_np
            else:
                y_true = np.append(y_true, label_np, axis=0)
                y_pre = np.append(y_pre, out_np, axis=0)

        acc = np.sum(y_pre == y_true) / y_true.shape[0]
        f1 = f1_score(y_true, y_pre, average='macro')
        # precision = precision_score(y_true, y_pre, average='macro')
        # recall = recall_score(y_true, y_pre, average='macro')

    return total_loss, f1, acc


if __name__ == '__main__':
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.model in ['DMBERT', 'CBiLSTM', 'DMCNN'], "只能选择 DMBERT / CBiLSTM / DMCNN 模型"
    if args.seed is not None:
        set_seed(args.seed)
    processor = Processor(args)
    train_features = processor.get_train_features()
    dev_features = processor.get_dev_features()

    # 训练dataset
    train_dataset = dataset.MyDataset(train_features)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size)
    # 验证dataset
    dev_dataset = dataset.MyDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size)

    # 开始训练
    model = get_class('models', args.model)(args)
    # print(model)
    # model = torch.load('./checkpoints/DMBERT.model')
    # model.eval()

    train(args, train_loader, dev_loader, model)
