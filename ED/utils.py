import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm


def build_optimizer(model, args, t_total):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and not any(nd in i for nd in no_decay))],
         'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if (not 'bert' in i and any(nd in i for nd in no_decay))],
         'lr': args.lr, 'weight_decay': 0.0},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and not any(nd in i for nd in no_decay))],
         'lr': args.bert_lr, 'weight_decay': args.weight_decay},
        {'params': [j for i, j in model.named_parameters() if ('bert' in i and any(nd in i for nd in no_decay))],
         'lr': args.bert_lr, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
        num_training_steps=t_total
    )
    return optimizer, scheduler

def evaluate(model, data_loader):
    y_true, y_pred = [], []
    total_loss = 0
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        for _, batch in tqdm(enumerate(data_loader)):
            true_label, out = model(batch)
            loss = loss_fn(out, true_label)
            total_loss += loss.item()
            pred = np.argmax(out.detach().cpu().numpy(), axis=1)
            label = np.argmax(true_label.detach().cpu().numpy(), axis=1)
            for i, l in enumerate(label):
                if l != 0:
                    y_true.append(l)
                    y_pred.append(pred[i])
        y_pred, y_true = np.array(y_pred), np.array(y_true)

        acc = np.sum(y_pred == y_true) / y_true.shape[0]
        f1 = f1_score(y_true, y_pred, average='macro')
    return total_loss, f1, acc