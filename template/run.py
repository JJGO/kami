import argparse
import json
from time import sleep

from kirin.util.gpu import restrict_GPU_keras

from experiment import Experiment


def jsonfile(file):
    with open(file, 'r') as f:
        s = json.load(f)
    return s


parser = argparse.ArgumentParser(description='Train DeepNet on VHM dataset on GPU using Keras')

parser.add_argument('-d', '--data', dest='data_params_f', type=json.loads, help='JSON string of Data parameters')
parser.add_argument('-m', '--model', dest='model_params_f', type=json.loads, help='JSON string of Model parameters')
parser.add_argument('-t', '--train', dest='train_params_f', type=json.loads, help='JSON string of Train parameters')
parser.add_argument('-C', '--config-file', dest='global_params', type=jsonfile, help='File of global parameters', default={})
parser.add_argument('-D', '--data-file', dest='data_params', type=jsonfile, help='File of Data parameters', default={})
parser.add_argument('-M', '--model-file', dest='model_params', type=jsonfile, help='File of Model parameters', default={})
parser.add_argument('-T', '--train-file', dest='train_params', type=jsonfile, help='File of Train parameters', default={})
# GPU params
parser.add_argument('-r', '--retain', dest='retain', action='store_true', default=False, help='Do no release GPU')
parser.add_argument('-p', '--path', dest='path', type=str, help='results_path', default=None)
parser.add_argument('-g', dest='gpuid', type=str, help='GPU id', default=None)
parser.add_argument('--mem', dest='memfrac', type=float, help='Fraction of memory', default=0)
parser.add_argument('--cpu', dest='use_cpu', action='store_true', default=False, help='Use the CPU instead of GPU')


if __name__ == '__main__':
    args = parser.parse_args()

    # Loading order is as follows
    # 1. global config file
    # 2. Data/Model/Train config files
    # 3. Data/Model/Train config strings
    # Last remains

    data_params = {}
    if 'data_params' in args.global_params:
        data_params.update(args.global_params['data_params'])
    data_params.update(args.data_params)
    data_params.update(args.data_params_f)

    model_params = {}
    if 'model_params' in args.global_params:
        model_params.update(args.global_params['model_params'])
    model_params.update(args.model_params)
    model_params.update(args.model_params_f)

    train_params = {}
    if 'train_params' in args.global_params:
        train_params.update(args.global_params['train_params'])
    train_params.update(args.train_params)
    train_params.update(args.train_params_f)

    restrict_GPU_keras(**args.__dict__)

    ex = Experiment(data_params=data_params,
                    model_params=model_params,
                    train_params=train_params,
                    path=args.path)

    ex.run()

    while args.retain:
        sleep(60)
