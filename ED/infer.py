import os
import numpy as np

import torch
import  torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from config import Config
from data_helper import DuEEDataset

def get_label(args):
    label = []
    with open(os.path.join(args.data_dir, 'labels.txt'), 'r', encoding='utf-8') as fr:
        line = fr.readline()
        while line:
            label.append(line.strip())
            line = fr.readline()
    return label

def infer_single(args, model, single_data, k):
    '''
    single_data format: {
        'text': TEXT,
        'trigger_list': [
            {
                'trigger': TRIGGER_WORD,
                'trigger_start_idx': START_IDX
            },
            ...
        ]
    }
    k: num of events to return
    '''
    trigger_list = single_data['trigger_list']
    data = DuEEDataset([single_data], args)
    loader = DataLoader(
        dataset=data, batch_size=args.eval_batch_size,
        sampler=SequentialSampler(data), drop_last=False
    )
    with torch.no_grad():
        model.eval()
        probs = []
        preds = []
        for batch in loader:
            _, out = model(batch)
            out = F.softmax(out, dim=1)
            out = out.detach().cpu().numpy()
            pred = np.argmax(out, axis=1)
            for i, p in enumerate(pred):
                if p != 0:
                    probs.append(out[i][p])
                    preds.append(p)
    results = []
    for i in range(len(probs)):
        results.append((probs[i], preds[i], trigger_list[i]['trigger']))
    results.sort(key=lambda x: x[0], reverse=True)
    results = results[:k]
    return results    
    

if __name__ == '__main__':
    args = Config()
    model = torch.load(os.path.join(args.output_dir, f'{args.model_name}.model'))
    if args.gpu >= 0:
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    id2label = get_label(args)

    single_data = {
        'text': '雀巢裁员4000人：时代抛弃你时，连招呼都不会打！',
        'trigger_list': [
            {'trigger': '裁员', 'trigger_start_index': 2},
            {'trigger': '抛弃', 'trigger_start_index': 10}
        ]
    }

    results = infer_single(args, model, single_data, 1)
    for i, (prob, pred, trigger) in enumerate(results):
        event_type = id2label[pred]
        print(f'Predict {i} # event_type: {event_type} , trigger: {trigger}, probability: {prob:.2%}')
