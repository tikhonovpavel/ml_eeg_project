import json
import inspect
import argparse
import os
import datetime

from train_val import start_training, DATE_FORMAT

# Create the parser
parser = argparse.ArgumentParser(description='A script that takes in arguments from the command line or a config file.')

# Define the arguments
parser.add_argument('--config', type=str, help='Path to the config file')

parser.add_argument('--experiment_description', type=str, help='Human readable description of the experiment')
parser.add_argument('--label_dir', type=str, help='Path to the label directory')
parser.add_argument('--img_dir', type=str, help='Path to the image directory')
parser.add_argument('--out_dir', type=str, help='Path to the output directory')
parser.add_argument('--log_dir', type=str, default='logs/', help='Path to the log directory')
parser.add_argument('--model', type=str, default='Unet3D', help='CNN architecture')
parser.add_argument('--loss_fn', type=str, default='L1Loss', help='Type of loss function')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--train_part', type=float, default=0.8, help='...')
parser.add_argument('--epochs', type=float, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--optimizer', type=str, default='Adam', help='Type of optimizer')
parser.add_argument('--decay', type=float, default=0.0, help='Decay rate for the optimizer')
parser.add_argument('--data_limit', type=float, default=4000, help='Dataset vplume')

# Parse the arguments
args = parser.parse_args()

# Load the config file if it exists
config = {}
if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)

# Replace the config properties if the correspondig command lines are specified
config.update({k: v for k, v in vars(args).items() if v is not None})  # {**args, **config}

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.isdir(args.log_dir):
    print('You should specify a directory for logs, not the file path')
    raise Exception()

log_number = max([int(x.split('_')[-1].split('.')[0]) for x in os.listdir(args.log_dir)], default=0) + 1
log_filename = f'log_{datetime.datetime.now().strftime(DATE_FORMAT)}_{log_number}.txt'
config['log_path'] = os.path.join(args.log_dir, log_filename)

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

# print(config.keys() - filtered_config.keys())
start_training(**filtered_config)
