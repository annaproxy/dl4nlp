#!/bin/bash
#SBATCH --job-name=gpt1
#SBATCH -t 18:06:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared_course

module load 2020
module load eb
module load Python/3.8.2-GCCcore-9.3.0 
module load cuDNN/8.0.3.33-gcccuda-2020a 
module load NCCL/2.7.8-gcccuda-2020a 


#python3.8 train_gpt.py --input bytes --batch_size 1024 --sequence_length 30 --input_dim 20000 --data_path "./data/wili-2018/x_bytes_train_clean.txt" --label_path "./data/wili-2018/y_train_clean.txt" --val_data_path "./data/wili-2018/x_bytes_val_clean.txt" --val_label_path "./data/wili-2018/y_val_clean.txt"
#python3.8 train_gpt.py --input bytes --batch_size 2048 --sequence_length 30 --input_dim 20000 --data_path "./data/wili-2018/x_bytes_train_clean.txt" --label_path "./data/wili-2018/y_train_clean.txt" --val_data_path "./data/wili-2018/x_bytes_val_clean.txt" --val_label_path "./data/wili-2018/y_val_clean.txt"
#python3.8 train_gpt.py --input bytes --batch_size 4096 --sequence_length 30 --input_dim 20000 --data_path "./data/wili-2018/x_bytes_train_clean.txt" --label_path "./data/wili-2018/y_train_clean.txt" --val_data_path "./data/wili-2018/x_bytes_val_clean.txt" --val_label_path "./data/wili-2018/y_val_clean.txt"
python3.8 predict.py --epochs 100 --input chars --model_checkpoint models/gpt/99_512_chars.pt --batch_size 512 --sequence_length 128 --input_dim 12300 --data_path "../lstm/data/wili-2018/x_train_sub_clean.txt" --label_path "../lstm/data/wili-2018/y_train_sub_clean.txt" --val_data_path "../lstm/data/wili-2018/x_val_sub_clean.txt" --val_label_path "../lstm/data/wili-2018/y_val_sub_clean.txt"
python3.8 predict.py --epochs 100 --input chars --model_checkpoint models/gpt/99_1024_chars.pt  --batch_size 1024 --sequence_length 64 --input_dim 12300 --data_path "../lstm/data/wili-2018/x_train_sub_clean.txt" --label_path "../lstm/data/wili-2018/y_train_sub_clean.txt" --val_data_path "../lstm/data/wili-2018/x_val_sub_clean.txt" --val_label_path "../lstm/data/wili-2018/y_val_sub_clean.txt"

python3.8 predict.py --prediction_type stochastic --epochs 100 --input chars  --model_checkpoint models/gpt/99_512_chars.pt --batch_size 512 --sequence_length 128 --input_dim 12300 --data_path "../lstm/data/wili-2018/x_train_sub_clean.txt" --label_path "../lstm/data/wili-2018/y_train_sub_clean.txt" --val_data_path "../lstm/data/wili-2018/x_val_sub_clean.txt" --val_label_path "../lstm/data/wili-2018/y_val_sub_clean.txt"
python3.8 predict.py --prediction_type stochastic --epochs 100 --input chars  --model_checkpoint models/gpt/99_1024_chars.pt --batch_size 1024 --sequence_length 64 --input_dim 12300 --data_path "../lstm/data/wili-2018/x_train_sub_clean.txt" --label_path "../lstm/data/wili-2018/y_train_sub_clean.txt" --val_data_path "../lstm/data/wili-2018/x_val_sub_clean.txt" --val_label_path "../lstm/data/wili-2018/y_val_sub_clean.txt"


#python3.8 train_gpt.py --input chars --epochs 100 --batch_size 2048 --sequence_length 128 --input_dim 12300 --data_path "../lstm/data/wili-2018/x_train_sub_clean.txt" --label_path "../lstm/data/wili-2018/y_train_sub_clean.txt" --val_data_path "../lstm/data/wili-2018/x_val_sub_clean.txt" --val_label_path "../lstm/data/wili-2018/y_val_sub_clean.txt"
