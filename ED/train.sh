python train.py --model="DMBERT" --bert_dir="/home/lixiang/bd/RoBERTa_zh_Large_PyTorch/" --data_dir="./data/" --log_dir="./logs/" --output_dir="./checkpoints/" --num_tags=66 --seed=123 --gpu_ids="0" --max_seq_len=256 --lr=2e-5 --other_lr=2e-4 --train_batch_size=20 --train_epochs=1 --eval_batch_size=32
