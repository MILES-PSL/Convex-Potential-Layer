from trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--save-dir', type=str, default="models_savedir")

# data args
parser.add_argument('--dataset', type=str, default="c10")
parser.add_argument('--norm_input', action='store_true')


# model args
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--depth-linear', type=int, default=7)
parser.add_argument('--num-channels', type=int, default=30)
parser.add_argument('--n-features', type=int, default=2048)
parser.add_argument('--conv-size', type=int, default=5)
parser.add_argument('--lln', action='store_true')

# optimization args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--margin', type=float, default=0.7)
parser.add_argument('--weight-decay', type=float, default=0.)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)

config = parser.parse_args()

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer()
    trainer.eval_final(eps=36. / 255)