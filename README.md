# EEG ML Project
## Report & presentation
[link1]

[link2]

## How to run
1. Load model weights and test set:

    `gdown --id ...`

    `gdown --id ...`

2. Run the script for prediction:

   `mkdir output`      

    `python training_script.py --model "VNet" --model_name "checkpoint" --out_dir "output" --predict_only 1`
   
   `python visualize.py`