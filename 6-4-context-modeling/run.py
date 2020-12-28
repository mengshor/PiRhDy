from settings import run_exp
import argparse

parser = argparse.ArgumentParser(description='input attributes')
parser.add_argument('--gpu_names', '-g',
                    default=None,
                    help='the ids of gpus, '
                         '\"None (default)\" for cpu running,'
                         '\"0,1\" for using multiple gpus')
parser.add_argument('--task_name', '-t',
                    default='next',
                    help='task names involve '
                         'next_phrase (next), '
                         'accompaniment_retrieval (acc), '
                         'genre (genre) '
                    )
parser.add_argument('--ratio', '-r',
                    default=100,
                    help='the ratio (0-100) of training dataset, e.g., 80% of data, -r=80')
parser.add_argument('--epoch_num', '-e',
                    default=100,
                    help='the epoch number')
parser.add_argument('--batch_size', '-b',
                    default=2000,
                    help='batch size')
parser.add_argument('--phrase_model', '-p',
                    default=None,
                    help='phrase model name')
args = parser.parse_args()

if __name__ == '__main__':
    try:
        print("run {} task on gpu {}".format(args.task_name, args.gpu_names))
        run_exp(gpu_names=args.gpu_names,
                task_name=args.task_name,
                ratio=float(args.ratio),
                epoch_num=int(args.epoch_num),
                batch_size=int(args.batch_size),
                phrase_model=args.phrase_model
                )

    except Exception as e:
        print('error: {}'.format(e))
