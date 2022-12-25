import json
import logging
import math
import os
from pprint import pformat

import datasets
import torch
import transformers
from accelerate import Accelerator
from fastcore.all import *
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler, set_seed

from models import get_auto_model
from utils.args import parse_args
from utils.data import get_dataloader_and_dataset, get_dataloader_and_test_dataset
from utils.postprocess import DedupList, isin, postprocess_gplinker
from utils.utils import get_writer, try_remove_old_ckpt, write_json

logger = logging.getLogger(__name__)


@torch.no_grad()
def test_one(
    args,
    model,
    dev_dataloader,
    accelerator,
    global_steps=0,
    threshold=0,
    write_predictions=True,
):
    model.eval()
    result, test_result = [], ""
    all_predictions = []
    for batch in tqdm(
        dev_dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Evaluating: ",
        leave=False,
    ):
        offset_mappings = batch.pop("offset_mapping")
        texts = batch.pop("text")
        outputs = model(**batch)[0]
        outputs_gathered = postprocess_gplinker(
            args,
            accelerator.gather(outputs),
            offset_mappings,
            texts,
            trigger=True,
            threshold=threshold,
        )
        all_predictions.extend(outputs_gathered)


    if write_predictions:
        pred_dir = os.path.join(args.output_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, f"test_result.json")
        f = open(pred_file, "w", encoding="utf-8")
    for pred_events, events, texts in zip(
        all_predictions,
        dev_dataloader.dataset.raw_data["events"],
        dev_dataloader.dataset.raw_data["text"],
    ):
        # R, T = DedupList(), DedupList()
        # # 事件级别
        # for event in pred_events:
        #     if any([argu[1] == "触发词" for argu in event]):
        #         R.append(list(sorted(event)))
        # for event in events:
        #     T.append(list(sorted(event)))
        # for event in R:
        #     if event in T:
        #         ex += 1
        # ey += len(R)
        # ez += len(T)
        # # 论元级别
        # R, T = DedupList(), DedupList()
        # for event in pred_events:
        #     for argu in event:
        #         if argu[1] != "触发词":
        #             R.append(argu)
        # for event in events:
        #     for argu in event:
        #         if argu[1] != "触发词":
        #             T.append(argu)
        # for argu in R:
        #     if argu in T:
        #         ax += 1
        # ay += len(R)
        # az += len(T)

        if write_predictions:
            event_list = DedupList()
            for event in pred_events:
                final_event = {
                    "event_type": event[0][0], "trigger":"", "trigger_start_index":"", "arguments": DedupList()}
                for argu in event:
                    if argu[1] != "触发词":
                        final_event["arguments"].append(
                            {"role": argu[1], "argument": argu[2]}
                        )
                    else:
                        final_event["trigger"] = argu[2]
                        final_event["trigger_start_index"] = argu[3].split(";")[0]
                event_list = [
                    event for event in event_list if not isin(event, final_event)
                ]
                if not any([isin(final_event, event) for event in event_list]):
                    event_list.append(final_event)

            result_event = {"text": texts, "event_list": event_list}
            l = json.dumps(result_event, ensure_ascii=False)
            f.write(l + "\n")
            test_result = test_result + l + "\n"
            result.append(result_event)

    if write_predictions:
        f.close()

    model.train()

    return result

def infer(data):
    args = parse_args()
    accelerator = Accelerator()
    if args.seed is not None:
        set_seed(args.seed)
    labels = []
    with open(os.path.join(args.file_path, "duee_event_schema.json"), "r", encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            t = l["event_type"]
            for r in ["触发词"] + [s["role"] for s in l["role_list"]]:
                labels.append((t, r))
    args.labels = labels
    args.num_labels = len(labels)
    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.pretrained_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = get_auto_model(args.model_type).from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.model_cache_dir,
        use_efficient=args.use_efficient,
    )
    # transfer single_data to file
    with open(os.path.join(args.file_path, "duee_test.json"), 'w', encoding='utf-8') as fw:
        for d in data:
            json.dump({'text': d['text'], 'id': '', 'event_list': []}, fw, ensure_ascii=False)
            fw.write('\n')
    test_dataloader = get_dataloader_and_test_dataset(
        args,
        tokenizer,
        labels,
        use_fp16=accelerator.use_fp16,
        text_column_name="text",
        label_column_name="events",
    )

    no_decay = ["bias", "LayerNorm.weight", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    model, optimizer, test_dataloader= accelerator.prepare(
        model, optimizer, test_dataloader
    )
    state_dict=torch.load('./outputs/last.pth')
    model.load_state_dict(state_dict)
    result = test_one(args, model, test_dataloader, accelerator, 0, 0, True)
    return result

def main():
    args = parse_args()
    accelerator = Accelerator()
    if args.seed is not None:
        set_seed(args.seed)
    labels = []
    with open(os.path.join(args.file_path, "duee_event_schema.json"), "r", encoding="utf-8") as f:
        for l in f:
            l = json.loads(l)
            t = l["event_type"]
            for r in ["触发词"] + [s["role"] for s in l["role_list"]]:
                labels.append((t, r))
    args.labels = labels
    args.num_labels = len(labels)
    tokenizer_name = (
        args.tokenizer_name
        if args.tokenizer_name is not None
        else args.pretrained_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = get_auto_model(args.model_type).from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.model_cache_dir,
        use_efficient=args.use_efficient,
    )
    test_dataloader = get_dataloader_and_test_dataset(
        args,
        tokenizer,
        labels,
        use_fp16=accelerator.use_fp16,
        text_column_name="text",
        label_column_name="events",
    )
    # (train_dataloader, dev_dataloader) = get_dataloader_and_dataset(
    #     args,
    #     tokenizer,
    #     labels,
    #     use_fp16=accelerator.use_fp16,
    #     text_column_name="text",
    #     label_column_name="events",
    # )

    no_decay = ["bias", "LayerNorm.weight", "norm"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    model, optimizer, test_dataloader= accelerator.prepare(
        model, optimizer, test_dataloader
    )

    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps
    # )
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # else:
    #     args.num_train_epochs = math.ceil(
    #         args.max_train_steps / num_update_steps_per_epoch
    #     )
    # args.num_warmup_steps = (
    #     math.ceil(args.max_train_steps * args.num_warmup_steps_or_radios)
    #     if isinstance(args.num_warmup_steps_or_radios, float)
    #     else args.num_warmup_steps_or_radios
    # )
    state_dict=torch.load('./outputs/last.pth')
    model.load_state_dict(state_dict)
    test_result = test_one(
                        args, model, test_dataloader, accelerator, 0, 0, True
                    )
    print(test_result)

if __name__ == "__main__":
    main()