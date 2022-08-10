# MS-STED

## Install 

You can use both conda and pip to install all the required dependencies.

```
git clone https://github.com/FrancescoGentile/MS-STED.git
cd MS-STED

# using conda (recommended)
conda env create -f conda_env.yml 
conda activate mssted

# using pip 
pip install -r requirements.txt
```

## How to

To generate a dataset, to train or test a model you will need to create one or more configuration files. See the folder ```config``` for some examples of config files.

### Generate a dataset

Create a (YAML) config file and under ```datasets``` add a list of all the datasets you want to generate. Then execute the following command:

```
python main.py --config /path/to/config/file.yaml --generate 
```

### Training
Create a (YAML) config file and add under ```trainings``` add a list of all the trainings you want to do. You need to add also the datasets, models, optimizers and LR schedulers you want to use.
Then execute the following command:

```
python main.py --config /path/to/config/file.yaml --train 
```

### Testing
TODO

```
python main.py --config /path/to/config/file.yaml --test 
```

### Distributed execution

You can speed up your training by using ```DistributedDataParallel```. To do this, create a distributed config file (like ```config/distributed.yaml```) and execute the following command:

```
python main.py (other commands) --distributed /path/to/distributed/config.yaml --rank n
```
If no distributed config file is passed, default distributed execution is created. If no cuda device is available, the main process will use the cpu for training. Instead, if N cuda devices are available, N processes will be created and each one will use one gpu for training. Thus, if more than one gpu is available but you want to use only one, you need to pass a config file. 

### Notes
1. You can also generate datasets, start trainings and tests with only one command. You simply need to create a (YAML) config file with all the required options and execute the following command:

```
python main.py --config /path/to/config/file.yaml --generate --train --test
```
The order of execution is always the same: dataset generation - training - test.

2. If an error occurs, you can add the ```--debug``` option so that the full stack trace will be added to the log. 

3. At the time of writing (08/2022), if you want to resume a training, you need to use at least pytorch 1.13 (nightly release) because of a problem in the current stable release (1.12).