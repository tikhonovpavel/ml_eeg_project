# EEG ML Project
## Report & presentation
materials/article_3D CNNs as solution for inverse EEG problem (Machine Learning 2023 Course.pdf

materials/presentation_3D CNN for inverse EEG problem.pdf

## How to run
* Load model weights and sample set:

    `gdown --id 1RlA_DInYxlvsQuDimHUYvtadrlnXrQgx`

    `gdown --id 1T_IpJMqIh78x4rSrQBiFZLz65-0pPk_-`

* Run the script for prediction:
   
  `mkdir output`

  `python preprocessing\unarchive_h5.py --h5_filename_to_unarchive "dense-162dip_parcell-64_GRID-64_paired_scale-False_DILATION-None-4000.h5"`    

  `python training_script.py --model "VNet" --model_name "VNet_trained_CrossEntropy_DILATION-None-4000_epoch-120.pt" --h5_file_path "dense-162dip_parcell-64_GRID-64_paired_scale-False_DILATION-None-4000.h5" --out_dir "output" --predict_only 1`
   
  `python plot/visualize.py --model_name "output/vnet_CrossEntropy_prediction.h5"`


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
│   ├── utils.py  # 
│   └── buildingblocks.py # realization of UNet architecture on pytorch
├── configs  # different run configs
│   ├── config_vnet_DILATION-2-predict.json  # 
│   ├── config_vnet_DILATION-2.json  # 
│   ├── config_vnet_DILATION-None-predict.json  # 
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
