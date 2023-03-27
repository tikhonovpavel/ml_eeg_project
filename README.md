# EEG ML Project
## Report & presentation
[link1]

[link2]

## How to run
* Load model weights and sample set:

    `gdown --id ...`

    `gdown --id ...`

* Run the script for prediction:
   
   `mkdir output`      

   `python training_script.py --model "VNet" --model_name "checkpoint.h5" --out_dir "output" --predict_only 1`
   
   `python plot/visualize.py --model_name "checkpoint.h5"`

## Train & test dataset

Download:

`gdown --id 1hRmT670aQNItDEGuEaUPvpphw6aDxNZu`

Unarchive:

`python preprocessing/unarchive_h5.py --filename "dense-162dip_parcell-64_GRID-64_paired_scale-False-2000.h5"`
