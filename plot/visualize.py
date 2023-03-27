import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.special import softmax
import random

augmented_data = False

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, help='Saved model checkpoint')
parser.add_argument('--num_of_plots', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.72)

args = parser.parse_args()


def deaugmentation(y, grid, AUG=2):
    y_deaugmented = np.zeros((grid, grid, grid))
    y_augmented = np.zeros((grid, grid, grid))

    mask = np.ones((AUG * 2 + 1, AUG * 2 + 1, AUG * 2 + 1)).tolist()
    for coord in y:
        y_augmented[tuple(coord.tolist())] = 1
    for coord in np.ndindex(y_augmented.shape):
        if y_augmented[coord[0] - AUG:coord[0] + AUG + 1,
           coord[1] - AUG:coord[1] + AUG + 1,
           coord[2] - AUG:coord[2] + AUG + 1].tolist() == mask: y_deaugmented[coord] = 1

    return np.array(np.where(y_deaugmented == 1)).T


def output_to_coords(array, threshold):
    array = array.squeeze()

    return np.array(np.where(array.squeeze() > threshold)).T


data = h5py.File(args.path, 'r')
keys = list(data.keys())

chosen_keys = random.choices(keys, k=args.num_of_plots)

for key in chosen_keys:

    y_true = data[key][3, 0]
    y_true = output_to_coords(y_true, 0.5)
    if augmented_data:
        y_true = deaugmentation(y_true, 64, 2)

    pred = np.stack((data[key][1, 0], data[key][2, 0]), axis=0)
    pred = np.transpose(pred, (1, 2, 3, 0))
    proba_one = softmax(pred, axis=3)

    proba_one = proba_one[:, :, :, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    proba_one[abs(proba_one) < args.threshold] = 0
    proba_one = proba_one / np.max(proba_one)
    x, y, z = np.array(np.nonzero(proba_one))
    # values = eeg[np.nonzero(eeg)]
    ax.scatter(x, y, z, c='darkblue', alpha=1)

    # y_true = y_true[np.nonzero(y_true)]
    x, y, z = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    ax.scatter(x, y, z, c='red', alpha=1)

    plt.show()
