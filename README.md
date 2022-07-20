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

## Create a config file 

See ```config/complete.yaml``` to see a complete config file with all the explanations.

## Generate a dataset

Create a (YAML) config file and under ```datasets``` add a list of all the datasets you want to generate. Then execute the following command:

```
python main.py --config /path/to/config/file.yaml --generate 
```

## Training
Create a (YAML) config file and add under ```trainings``` add a list of all the trainings you want to do. You need to add also the datasets, models, optimizers and LR schedulers you want to use.
Then execute the following command:

```
python main.py --config /path/to/config/file.yaml --train 
```

## Testing
TODO

```
python main.py --config /path/to/config/file.yaml --test 
```

## Notes
You can also generate datasets, start trainings and tests with only one command. You simply need to create a (YAML) config file with all the required options and execute the following command:

```
python main.py --config /path/to/config/file.yaml --generate --train --test
```

The order of execution is always the same: dataset generation - training - test.

If an error occurs, you can add the ```--debug``` option so that the full stack trace will be added to the log. 