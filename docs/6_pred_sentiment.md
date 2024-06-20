# Print Sentiment Prediction Examples
Here, we provide guides for printing sentiment prediction results of your data.

### 1. Sentiment Prediction
#### 1.1 Arguments
There are several arguments for running `src/run/sentiment_prediction.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to predict. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/`.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `validation`) Choose one of [`train`, `validation`, `test`].
* [`-n`, `--result_num`]: (default: `10`) The number of random data to visualize.

#### 1.2 Command
`src/run/sentiment_prediction.py` file is used to predict sentiment results of the model with the following command:
```bash
python3 src/run/sentiment_prediction.py --resume_model_dir ${project}/${name}
```