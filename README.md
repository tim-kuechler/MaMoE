This is the repository for my master thesis "Benchmarking and Advancing Diffusion Model Fine-Tuning on Narrow-Domain 
Datasets." that you can find here [here](https://heibox.uni-heidelberg.de/f/101047ddb0c04c78ab8c/?dl=1).

On the `main` branch all code for replicating the results from the thesis is provided.
In the branch `dm-training` you can find a version of the code without MaMoe, that can be
used to fine-tune diffusion models.

## Project Structure
Configs are located in `configs/` and are using [Hydra](https://hydra.cc/docs/intro/) as a config management system.
Hydra's configs are build hierarchically in the sense that any config can be overwritten by nested configs. For example
`train.yaml` defines the root config, that loads the default configs from each of the subfolders (`callbacks/`, `data/`
, ...). For each of the subconfigs again there can be multiple versions overwriting the default version. For example
in `debug`, there exists a default config and others that can be specified via the command line.

Typically (if you don't want to change default settings globally), all you ever need to change/create are new dataset 
configs in `data/` (similar to `cityscapes.yaml`) and new experiments in `experiment/`, similar to 
`experiment_demo.yaml`.

The most important files are `lightning.py`, where the training loop is defined and `unet.py` where model
initialization and the model forward pass are handled.

All MaMoE related code can be found under `models/mamae/.`


```
MaMoE/
├── configs/            All configs are saved here
│   ├── callbacks/      Configs for callbacks (checkpoint saving, image logging, metric calculation)      
│   ├── data/           Configs defining the dataset 
│   ├── debug/          Configs to set the training to debug mode
│   ├── experiment/     The experiment configs that overwrite all other configs
│   ├── hydra/          Hydra configs
│   ├── logger/         Logging configs (esp. wandb)
│   ├── model/          Configs for the model, training loop and UNet
│   ├── trainer/        Configs for PyTorch Lightning's trainer
│   └── train.yaml      The root config
├── dmt/
│   ├── data/
│   │   ├── dataset.py          A custom PyTorch dataset loading data structured as described below
│   │   ├── lightning_data.py   PyTorch Lightning's DataModule responsible for initializing the dataloaders during training
│   │   └── utils.py            Data utils.
│   ├── lightning_modules/
│   │   └── lightning.py        PyTorch Lightning's LightningModule. The main training loop, optimizer loading, ... is located here
│   ├── models/
│   │   ├── diffusion/          Code related to Huggingface Diffusion Models (UNet and VAE)
│   │   │   ├── unet.py         Contains code to initialize a Huggingface UNet and a Wrapper around it handling the inputs to the UNet
│   │   │   └── vae.py          Contains code to initialize a Huggingface VAE and a wrapper around it making usage easier
│   │   └── mamoe/              Code related to the MaMoE method
│   │       ├── cross_attention/                    Code related to Spatial Prompting with Masked Cross Attentions
│   │       │   ├── load_cross_attention_data.py    Loads the panoptic map and instance prompts and converts both into a instance based attention masks 
│   │       │   └── masked_attention.py             The implementation of the masked attention mechanism
│   │       │── moe/                                Code related to class-aware Mixture of Experts
│   │       │   └── load_moe_data.py                Loads panoptic maps to create expert maps
│   │       │   └── moe.py                          The implementanion of class-aware MoE
│   │       └── utils.py                            Utils for MaMoE
│   └── utils/          
│   │   ├── eval/               Contains evaluation and sampling code
│   │   ├── callbacks.py        Custom callbacks for image logging, score computation, ...        
│   │   ├── extra_utils.py
│   │   ├── instantiators.py
│   │   ├── logging_utils.py    Multi-GPU logger
│   │   ├── model_utils.py      
│   │   ├── task_utils.py
│   └── train.py                The starting point for training
├── environment.yaml            Conda environment file
├── pyproject.toml              File for setting up the repo as a python module
└── README.md                   This file
```

## Data Structure
The data that is to be loaded by the dataset in `dataset.py` is expected to have the following structure:

```
<dataset_root_dir>/     Is defined for example in configs/data/cityscapes.yaml at dataset_dir
├── train/              Training images
├── val/                Validation images
└── test/               Test images
```

Images are expected to have one number somewhere in the filename (e.g. `000001`) in order to sort them correctly.

Panoptic maps are expected to have the following structure:
```
<panoptic_map_dir>/     Is defined for example in configs/experiment/base_experiment.yaml at model -> panoptic_map_dir:
├── train/              Training panoptic maps
├── val/                Validation panoptic maps
└── test/               Test panoptic maps
```

Panoptic maps are expected to have exactly the same filename as their corresponding RGB images.

Global prompts (i.e. normal prompts equivalent to normal diffusion model training) are e.g. provided in 
`configs/data/cityscapes.yaml` at `<train/val/test>_dataset: -> prompts_file: <prompts_file_path>.csv`. The csv containing
the prompts is expected to have a header with two columns: `img_name` and `prompt`, where `img_name` matches image
names from (only file name, not path) RGB training images in `<dataset_root_dir>/...`, and promt is the prompt. To write
our csv file we used `csv.DictWriter`.

Instance prompts can be added optionally to e.g. `configs/data/cityscapes.yaml` at `<train/val/test>_dataset: -> 
instance_prompts_file: <instance_file_path>.json`. Instance prompts are expected to be a python dictionary
that was saved using `json`. The structure of that dictionary is
```
<img_name (as string)>:
    <instance_id (as int, matching instance id in panoptic map)>: <instance_prompt (as string)>
    ...
 ...
```
## Train

Before training you need to prepare a few things:
1. Create a conda environment using `conda env create -f environment.yaml`
2. Structure your image data as described above. 3
3. Generate panoptic maps and structure them as described above. I used [MIC](https://github.com/lhoyer/MIC) to generate
    coarse semantic maps and [YOLO v11](https://github.com/ultralytics/ultralytics) to generate instance masks.
    I then combined both into one panoptic map. Panoptic maps are encoded into RGB channels. RG are the instance id 
    channels, B the class id channel. Refer to `models/`
4. Generate global prompts and define them in your dataset config as described above. I used the Batch-API from 
    ChatGPT-4o-mini to caption the images. Instructions to the LLM are provided in the appendix of my thesis.
5. Optional: Generate prompts for every instance in the panoptic map and write them to a json as described above.
    I also used ChatGPT-4o-mini with OpenAI's Batch-API to generate these prompts. I cut out the thing objects (car,
    bus, ...) by their bounding box and colored any region not belonging to the object black. Instructions to the LLM 
    are provided in the appendix of my thesis.
6. Define the `log_dir` in `configs/train.yaml`. This is the root directory where all your experiments, checkpoints, ...
    will be saved.
7. Sometimes the program crashes due to too many open file handles. If something like this happens run e.g. 
    `ulimit -n 500000` to increase the limit.
8. Define your dataset. Create a new `configs/data/<your_dataset_name>.yaml` for your dataset. Take `cityscapes.yaml` as
    an example.
9. Create your own experiment config in `configs/experiment/<your_experiment_name>.yaml`. See `experiment_demo.yaml` as
    an example. For multi-gpu training overwrite the `gpus:` variable that is set to 1 in `base_experiment.yaml`.
10. Install this repository into your conda environment with `pip install -e .`.
11. Start the training with e.g. 
    `CUDA_VISIBLE_DEVICES=0 python train.py data=<your_dataset_name> experiment=<your_experiment_name>`
12. If you want to debug the program consider adding `debug=<default/data_parallel/limit/...>` as defined in 
    `configs/debug/` to your command.

## Inference
Currently, there is no script for inference provided. Technically the model checkpoints can be 
loaded with the normal HuggingFace SD2.1 pipeline. However, the inputs must be adapted to match the
the data my dataset returns and as they are expected by mamoe in `models/mamoe`.

During training there will be samples generated and saved to disk or wandb. These samples where used to
generate the scores in the thesis and are presented in the thesis.