import argparse
import time

""" MAIN """
parser = argparse.ArgumentParser(description='FCA')
parser.add_argument('--type', default=3, type=int)
parser.add_argument('--seed', default=2025, type=int)
parser.add_argument('--iters', default=100, type=int)
parser.add_argument('--numItermax', default=1000000, type=int)
parser.add_argument('--numThreads', default=5, type=int)
parser.add_argument('--epsilon', default=0.0, type=float)
parser.add_argument('--lr', default=5e-2, type=float)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--data_name', default='Adult', type=str)
parser.add_argument('--l2_normalize', action='store_true')
parser.add_argument('--gradient_descent', action='store_true')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--max_iter', default=300, type=int)
parser.add_argument('--iters_inner', default=1, type=int)
parser.add_argument('--K', default=-1, type=int)

args = parser.parse_args()
print('='*20)
for key, value in vars(args).items():
    print(f'\t [{key}]: {value}')

if __name__ == "__main__":
    if args.epsilon == 0:
        from src.FCA import FCA
        runner = FCA.runner
    else:
        from src.FCA_C import FCA_C
        runner = FCA_C.runner
    runner(args)
    print('='*20)