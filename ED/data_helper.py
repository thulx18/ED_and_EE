import os, json, random

import jieba
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer


class DuEEDataset(Dataset):
    def __init__(self, data, args):
        self.num_class = args.num_class
        self.max_len = args.max_seq_len
        self.bert_dir = "/home/lixiang/bd/RoBERTa_zh_Large_PyTorch/"
        # get label map
        self.label2id = {}
        label_path = "/home/lixiang/ED_and_EE/ED/data/labels.txt"
        if not os.path.exists(label_path):
            labels = []
            with open("/home/lixiang/ED_and_EE/ED/data/raw_data/event_schema.json", 'r', encoding='utf-8') as fr:
                line = fr.readline()
                while line:
                    d = json.loads(line.strip())
                    labels.append(d['event_type'])
                    line = fr.readline()
            with open(label_path, 'w', encoding='utf-8') as fw:
                for label in labels:
                    fw.write(label + '\n')
            for i, label in enumerate(labels):
                self.label2id[label] = i
        else:
            with open(label_path, 'r', encoding='utf-8') as fr:
                idx = 0
                line = fr.readline()
                while line:
                    self.label2id[line.strip()] = idx
                    idx += 1
                    line = fr.readline()
        # get features
        examples = self.get_examples(data)
        self.features = self.get_features(examples)

    def get_examples(self, raw_data):
        examples = []
        for raw in raw_data:
            text = raw['text']
            if 'event_list' in raw:
                event_list = raw['event_list']
                last_triggerL = 0
                for event in event_list:
                    event_type = event['event_type']
                    trigger = event['trigger']
                    offset = len(trigger) - len(trigger.lstrip())
                    trigger = trigger.lstrip()
                    trigger_start_index = event['trigger_start_index'] + offset
                    trigger_end_index = trigger_start_index + len(trigger) + offset

                    tokensL = jieba.lcut(text[:trigger_start_index])
                    tokensR = jieba.lcut(text[trigger_end_index:])
                    tokens = tokensL + [trigger] + tokensR
                    triggerL = len(tokensL)
                    triggerR = triggerL + 1

                    examples.append({
                        'text': text,
                        'tokens': tokens,
                        'triggerL': triggerL,
                        'triggerR': triggerR,
                        'event_type': event_type
                    })

                    # add negative sample
                    if len(tokensL[last_triggerL:]) <= 0:
                        continue
                    trigger = random.sample(tokensL[last_triggerL:], 1)[0]
                    last_triggerL = triggerL
                    trigger_start_index = text.find(trigger)
                    if trigger_start_index == -1:
                        continue
                    trigger_end_index = trigger_start_index + len(trigger)
                    tokensL = jieba.lcut(text[:trigger_start_index])
                    tokensR = jieba.lcut(text[trigger_end_index:])
                    tokens = tokensL + [trigger] + tokensR
                    triggerL = len(tokensL)
                    triggerR = triggerL + 1

                    examples.append({
                        'text': text,
                        'tokens': tokens,
                        'triggerL': triggerL,
                        'triggerR': triggerR,
                        'event_type': event_type
                    })
            else:
                # Test Data, no event_list, but has trigger_list
                for trigger in raw['trigger_list']:
                    trigger_word = trigger['trigger']
                    offset = len(trigger_word) - len(trigger_word.lstrip())
                    trigger_word = trigger_word.lstrip()
                    trigger_start_index = int(trigger['trigger_start_index']) + offset
                    trigger_end_index = trigger_start_index + len(trigger_word) + offset

                    tokensL = jieba.lcut(text[:trigger_start_index])
                    tokensR = jieba.lcut(text[trigger_end_index:])
                    tokens = tokensL + [trigger_word] + tokensR
                    triggerL = len(tokensL)
                    triggerR = triggerL + 1

                    examples.append({
                        'text': text,
                        'tokens': tokens,
                        'triggerL': triggerL,
                        'triggerR': triggerR
                    })
                
        return examples
    
    def get_features(self, examples):
        tokenizer = BertTokenizer.from_pretrained(self.bert_dir)
        features = []
        for example in examples:
            tokens, triggerL, triggerR = example['tokens'], example['triggerL'], example['triggerR']
            textL = tokenizer.tokenize("".join(tokens[:triggerL]))
            textR = ['[unused0]']
            textR += tokenizer.tokenize(''.join(tokens[triggerL:triggerR]))
            textR += ['[unused1]']
            textR += tokenizer.tokenize(''.join(tokens[triggerR:]))
            maskL = [1. for _ in range(0, len(textL) + 1)] + [0. for _ in range(0, len(textR) + 1)]
            maskR = [0. for _ in range(0, len(textL) + 1)] + [1. for _ in range(0, len(textR) + 1)]

            if len(maskL) > self.max_len:
                maskL = maskL[:self.max_len]
            if len(maskR) > self.max_len:
                maskR = maskR[:self.max_len]

            inputs = tokenizer.encode_plus(
                textL + textR, add_special_tokens = True,
                max_length=self.max_len, padding='max_length', truncation=True,
                return_token_type_ids = True, return_overflowing_tokens=True
            )
            # padding
            pad_len = self.max_len - len(maskL)
            maskL = maskL + [0.] * pad_len
            maskR = maskR + [0.] * pad_len
            # convert event_type to one-hot
            if 'event_type' in example:
                event_type = [0.] * self.num_class
                event_type[self.label2id[example['event_type']]] = 1.
                features.append({
                    'input_ids': torch.LongTensor(inputs['input_ids']),
                    'attention_mask': torch.LongTensor(inputs['attention_mask']),
                    'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
                    'maskL': torch.tensor(maskL),
                    'maskR': torch.tensor(maskR),
                    'event_type': torch.tensor(event_type)
                })
            else:
                features.append({
                    'input_ids': torch.LongTensor(inputs['input_ids']),
                    'attention_mask': torch.LongTensor(inputs['attention_mask']),
                    'token_type_ids': torch.LongTensor(inputs['token_type_ids']),
                    'maskL': torch.tensor(maskL),
                    'maskR': torch.tensor(maskR)
                })
        return features

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]


def create_dataloader(args, data_path, data_type='eval'):
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as fr:
        line = fr.readline()
        while line:
            raw_data.append(json.loads(line.strip()))
            line = fr.readline()
    data_set = DuEEDataset(raw_data, args)
    is_train = (data_type == 'train')
    loader = DataLoader(
        dataset=data_set,
        batch_size=args.train_batch_size if is_train else args.eval_batch_size,
        sampler=RandomSampler(data_set) if is_train else SequentialSampler(data_set),
        drop_last=is_train
    )

    return loader