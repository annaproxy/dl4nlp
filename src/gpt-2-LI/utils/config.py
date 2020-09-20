import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=42)

    parser.add_argument('--input_dim', type=int, default=12300)
    parser.add_argument('--embedding_dim', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--model_checkpoint', type=str, default=None)

    parser.add_argument('--data_path', type=str, default="./data/wili-2018/x_train_sub.txt")
    parser.add_argument('--label_path', type=str, default="./data/wili-2018/y_train_sub.txt")

    parser.add_argument('--val_data_path', type=str, default="./data/wili-2018/x_val_sub.txt")
    parser.add_argument('--val_label_path', type=str, default="./data/wili-2018/y_val_sub.txt")

    args = parser.parse_args()
    print(args)
    with open("config.txt", 'w') as file:
        file.write(str(args))
    return args
