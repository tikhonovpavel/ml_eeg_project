import json
import inspect
import argparse
import os
import datetime
from distutils.util import strtobool

from train_val import start_training, DATE_FORMAT
from predict_set import predict_set
import sys

import wandb
    

# Create the parser
parser = argparse.ArgumentParser(description='A script that takes in arguments from the command line or a config file.')

# Define the arguments
parser.add_argument('--config', type=str, help='Path to the config file')

parser.add_argument('--experiment_description', type=str, help='Human readable description of the experiment')
parser.add_argument('--h5_file_path', type=str, help='Path to the unarchived h5 file')
parser.add_argument('--out_dir', type=str, help='Path to the output directory')
parser.add_argument('--log_dir', type=str, help='Path to the log directory')

parser.add_argument('--model', type=str, help='CNN architecture')
parser.add_argument('--loss_fn', type=str, help='Type of loss function')
parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
parser.add_argument('--gamma', type=float, help='Scheduler parameter')
parser.add_argument('--train_part', type=float, help='Proportion of the dataset to use for training')
parser.add_argument('--epochs', type=float, help='Number of epochs')
parser.add_argument('--batch_size', type=int, help='Batch size for training')
parser.add_argument('--optimizer', type=str, help='Type of optimizer')
parser.add_argument('--decay', type=float, help='Decay rate for the optimizer')

parser.add_argument('--predict_only', type=lambda x:bool(strtobool(x)), default=False, nargs='?', help='Perform only prediction')
parser.add_argument('--model_name', type=str, help='Path to saved model checkpoint')
parser.add_argument('--set_size', type=int, help='How many images to predict as examples')

parser.add_argument('--debug_launch', type=lambda x: bool(strtobool(x)), default=False, help='Set to true for debugging')
parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False, help='Set to true to enable wandb logging')

# Parse the arguments
args = parser.parse_args()

# Load the config file if it exists
config = {}
if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f'Config {args.config} loaded')
else:
    print('No config specified')

# Replace the config properties if the corresponding command lines are specified
config.update({k: v for k, v in vars(args).items() if v is not None})  # {**args, **config}

if config['predict_only'] == True:
    prediction_func_signature = inspect.signature(predict_set)
    filtered_config = {k: config[k] for k in prediction_func_signature.parameters.keys() if k in config.keys()}
    predict_set(**filtered_config)
    sys.exit()
    
if not os.path.exists(config['log_dir']):
    os.makedirs(config['log_dir'])

if not os.path.isdir(config['log_dir']):
    print('You should specify a directory for logs, not the file path')
    raise Exception()

log_number = max([int(x.split('_')[-1].split('.')[0]) for x in os.listdir(config['log_dir'])], default=0) + 1
log_filename = f'log_{datetime.datetime.now().strftime(DATE_FORMAT)}_{log_number}.txt'
config['log_path'] = os.path.join(config['log_dir'], log_filename)

print('Start training with the following configuration:')
print()
for k, v in config.items():
    print(f'{k}: {v}')
print('-' * 50)

training_func_signature = inspect.signature(start_training)
filtered_config = {k: config[k] for k in training_func_signature.parameters.keys()}

with open(config['log_path'], 'a') as log:
    log.write('Start training with the following configuration:\n\n')
    for k, v in config.items():
        log.write(f'{k}: {v}\n')
    log.write('-' * 50 + '\n')


if config['use_wandb']:
    wandb.init(project='ml-eeg', config=config)


start_training(**filtered_config)


