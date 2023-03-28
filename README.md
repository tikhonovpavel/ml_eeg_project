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

## Repository structure
``` bash
EEG ML Project
├── report  # report deliverables
│   ├── presentation.pdf
│   └── report.pdf
├── plot
│   └── visualize.py
├── predictor  # contains models schemes and neutron files
│   └── buildingblocks.py # realization of UNet architecture on pytorch
├── configs  # different run configs
│   └── config_vnet_DILATION-2-predict.json  # 
│   └── config_vnet_DILATION-2.json  # 
│   └── config_vnet_DILATION-None-predict.json  # 
│   └── config_vnet_DILATION-None.json  # 
├── preprocessing
│   ├── coords_to_voxels.ipynb  # 
│   ├── data_to_h5_dtype.py  # 
│   └── unarchive_h5.py  # unarchives the dataset in h5 format
├── CustomImageDataset.py  #
├── metric_for_dataset.ipynb  # calcules the metric of the model preiction
├── models.py  #
├── predict_set.py  #
├── requirements.txt
└── README.md
```