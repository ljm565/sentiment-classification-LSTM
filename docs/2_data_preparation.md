# Data Preparation
Here, we will proceed with a LSTM sentiment classification model training tutorial using the [IMDb](http://ai.stanford.edu/~amaas/data/sentiment/) dataset by default.
Please refer to the following instructions to utilize custom datasets.

### 1. IMDb
If you want to train on the IMDb dataset, simply set the `IMDb_train` value in the `config/config.yaml` file to `True` as follows.
```yaml
IMDb_train: True       
IMDb:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train your custom dataset, set the `IMDb_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
```yaml
IMDb_train: False       
IMDb:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
