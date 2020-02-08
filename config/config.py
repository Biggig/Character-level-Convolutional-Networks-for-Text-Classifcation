import argparse

def get_args():  # 初始化参数
    parser = argparse.ArgumentParser(
        """Character-level Convolutional Networks for Text Classification (https://arxiv.org/pdf/1509.01626.pdf)""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--train_data", type=str,
                        default="data\\ag_news\\raw\\train.csv")
    parser.add_argument("--test_data", type=str,
                        default="data\\ag_news\\raw\\test.csv")
    parser.add_argument("--model_folder", type=str, default="model/model/")
    parser.add_argument("--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\|_@#$%^&*~`+=<>()[]{}""")
    parser.add_argument("--maxlen", type=int, default=1014)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--solver", type=str,
                        default="sgd", help="'agd' or 'adam'")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=10,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gpuid", type=int, default=0,
                        help="select gpu (-1 if cpu)")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--existed", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="model_epoch_")
    parser.add_argument("--train_output", type=str, default="output/train_output.txt")
    parser.add_argument("--test_output", type=str, default="output/test_output.txt")
    args = parser.parse_args()
    return args
