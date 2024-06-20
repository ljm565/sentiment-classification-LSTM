# Data Preparation
여기서는 기본적으로 [IMDb](http://ai.stanford.edu/~amaas/data/sentiment/) 데이터셋을 활용하여 LSTM 감성 분류 모델 학습 튜토리얼을 진행합니다.
Custom 데이터를 이용하기 위해서는 아래 설명을 참고하시기 바랍니다.

### 1. IMDb
IMDb 데이터를 학습하고싶다면 아래처럼 `config/config.yaml`의 `IMDb_train`을 `True`로 설정하면 됩니다.
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
만약 custom 데이터를 학습하고 싶다면 아래처럼 `config/config.yaml`의 `IMDb_train`을 `False`로 설정하면 됩니다.
다만 `src/utils/data_utils.py`에 custom dataloader를 구현해야할 수 있습니다.
```yaml
IMDb_train: True       
IMDb:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```