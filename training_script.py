import json
import inspect
import argparse

from Unet3D_AlAu_normal_enchanced import start_training

# Create the parser
parser = argparse.ArgumentParser(description='A script that takes in arguments from the command line or a config file.')

# Define the arguments
parser.add_argument('--config', type=str, help='Path to the config file')

parser.add_argument('--experiment_description', type=str, help='Human readable description of the experiment')
parser.add_argument('--label_dir', type=str, help='Path to the label directory')
parser.add_argument('--img_dir', type=str, help='Path to the image directory')
parser.add_argument('--out_dir', type=str, help='Path to the output directory')
parser.add_argument('--log_path', type=str, help='Path to the log file')
parser.add_argument('--loss_fn', type=str, default='l1loss', help='Type of loss function')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--train_part', type=float, default=0.8, help='...')
parser.add_argument('--epochs', type=float, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--optimizer', type=str, default='adam', help='Type of optimizer')
parser.add_argument('--decay', type=float, default=0.0, help='Decay rate for the optimizer')

# Parse the arguments
args = parser.parse_args()

# Load the config file if it exists
config = {}
if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)

# Replace the config properties if the correspondig command lines are specified
config.update({k: v for k, v in vars(args).items() if v is not None})  # {**args, **config}

training_func_signature = inspect.signature(start_training)

print('Start training with the following configuration:')
print()
for k, v in config.items():
    print(f'{k}: {v}')
print('-' * 50)

filtered_config = {k: config[k] for k in training_func_signature.parameters.keys()}
# print(config.keys() - filtered_config.keys())
start_training(**filtered_config)
