#!/bin/bash
#SBATCH --job-name=lstm2
#SBATCH -t 18:06:00
#SBATCH -N 1
#SBATCH --partition=gpu_shared_course

module load 2020
module load eb
module load Python/3.8.2-GCCcore-9.3.0 
module load cuDNN/8.0.3.33-gcccuda-2020a 
module load NCCL/2.7.8-gcccuda-2020a 




#python3.8 train.py --epochs 100 --hidden_dim 768 --batch_size 1024 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 32

#python3.8 train.py --epochs 100 --hidden_dim 768 --batch_size 2048 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 32

#python3.8 train.py --epochs 100 --hidden_dim 768 --batch_size 4096 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 32

python3.8 predict.py --prediction_type deterministic --model_checkpoint models/LSTM/99_1024_chars_48.pt  --epochs 100 --hidden_dim 768 --batch_size 1024 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 48

python3.8 predict.py  --prediction_type deterministic  --model_checkpoint models/LSTM/99_2048_chars_48.pt --hidden_dim 768 --epochs 100 --batch_size 2048 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 48

python3.8 predict.py  --prediction_type deterministic  --model_checkpoint models/LSTM/99_4096_chars_48.pt --hidden_dim 768 --epochs 100 --batch_size 4096 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 48

python3.8 predict.py   --prediction_type deterministic --model_checkpoint models/LSTM/99_1024_chars_64.pt --hidden_dim 768 --epochs 100 --batch_size 1024 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 64

python3.8 predict.py  --prediction_type deterministic  --model_checkpoint models/LSTM/99_2048_chars_64.pt --hidden_dim 768 --epochs 100 --batch_size 2048 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 64

python3.8 predict.py  --prediction_type deterministic  --model_checkpoint models/LSTM/99_4096_chars_64.pt --hidden_dim 768 --epochs 100 --batch_size 4096 --input chars --input_dim 12300 --data_path "./data/wili-2018/x_train_sub_clean.txt" --label_path "./data/wili-2018/y_train_sub_clean.txt" --val_data_path "./data/wili-2018/x_val_sub_clean.txt" --val_label_path "./data/wili-2018/y_val_sub_clean.txt" --sequence_length 64


