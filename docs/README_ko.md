# Sentiment Classification LSTM

## Introduction
IMDb 영화 리뷰 데이터 바탕으로 LSTM 모델을 사용하여 긍부정 감성 분류 모델을 제작합니다.
본 코드에서는 attention의 유무에 따른 감성 분류 모델을 제작할 수 있습니다.
LSTM 기반 감성 분류 모델과 이 모델의 attention에 대한 설명은 [Sequence-to-Sequence (Seq2Seq) 모델과 Attention](https://ljm565.github.io/contents/RNN2.html)을 참고하시기 바랍니다.
<br><br><br>

## Supported Models
### Bidirectional LSTM and Attention
* `nn.LSTM`을 사용한 bidirectional LSTM이 구현되어 있습니다.
* `config/config.yaml`에서 모델 attention 기법 여부를 설정할 수 있습니다.
<br><br><br>

## Supported Tokenizer
### Custom Word Tokenizer
* 단어별 attention 가시화를 위해 단어 기준으로 토큰화 합니다.
<br><br><br>

## Base Dataset
* 튜토리얼로 사용하는 기본 데이터는 [IMDb](http://ai.stanford.edu/~amaas/data/sentiment/) 데이터입니다.
* `config/config.yaml`에 학습 데이터의 경로를 설정하여 사용자가 가지고 있는 custom 데이터도 학습 가능합니다.
다만 `src/utils/data_utils.py`에 custom dataloader 코드를 구현해야할 수도 있습니다.
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>


## Project Tree
본 레포지토리는 아래와 같은 구조로 구성됩니다.
```
├── configs                           <- Config 파일들을 저장하는 폴더
│   └── *.yaml
│
└── src      
    ├── models
    |   └── model.py                  <- LSTM 모델 파일
    |
    ├── run                   
    |   ├── sentiment_prediction.py   <- 예측 결과 프린트 코드
    |   ├── train.py                  <- 학습 실행 파일
    |   ├── validation.py             <- 학습된 모델 평가 실행 파일
    |   └── vis_attention.py          <- 단어별 attention 비율 가시화 코드
    |
    ├── tools                   
    |   ├── tokenizers
    |   |    └── word_tokenizer.py    <- 단어 토크나이저 파일
    |   |
    |   ├── imdb_downloader.py        <- IMDb 데이터 다운로드 파일
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class 파일
    |
    ├── trainer                 
    |   ├── build.py            <- Dataset, dataloader 등을 정의하는 파일
    |   └── trainer.py          <- 학습, 평가, accuracy 계산 class 파일
    |
    └── uitls                   
        ├── __init__.py         <- Logger, 버전 등을 초기화 하는 파일etc.
        ├── data_utils.py       <- Custom dataloader 파일
        ├── filesys_utils.py       
        ├── func_utils.py       
        └── training_utils.py     
```
<br><br>


## Tutorials & Documentations
LSTM 감성 분류 모델 학습을 위해서 다음 과정을 따라주시기 바랍니다.
1. [Getting Started](./1_getting_started_ko.md)
2. [Data Preparation](./2_data_preparation_ko.md)
3. [Training](./3_trainig_ko.md)
4. ETC
   * [Evaluation](./4_model_evaluation_ko.md)
   * [Attention Visualization](./5_vis_attn_ko.md)
   * [Print Sentiment Prediction Results](./6_pred_sentiment_ko.md)
<br><br><br>

## Training Results
### Sentiment Classification Results of LSTM w/ Attention
* Loss History<br>
<img src="docs/figs/output1.jpg" width="80%"><br><br>

* Accuracy History<br>
<img src="docs/figs/output3.jpg" width="80%"><br><br>
Model w/ attention: 0.884720<br><br>

* Samples of sentiment prediction (ground truth: 1.0 (positive), 0.0 (negative))<br>
    ```
    i saw [UNK] on broadway and liked it a great deal i don't know what happened with the film version because it was dreadful perhaps some dialogue that works on stage just sounds incoherent on screen anyway i couldn't wait for this film to be over the acting is universally over the top only kevin spacey has it together and he seems like he knows he's in a bad movie and can't wait to get out
    ******************************************
    It is positive with a probability of 0.216
    ground truth: 0.0
    ******************************************


    how do these guys keep going they're about 50 years old each and act as if they're only 30 they play 3 hours of music at every concert and barely break a sweat this dvd is their first concert in [UNK] brazil although the people don't speak english they try to [UNK] the words to the most famous rush songs and try to sing a foreign language at the concert with their best friends from tom [UNK] to the spirit of radio this concert dvd will keep you in the chair not wanting to pause or move away from the classics that you've listened to when you were young this is their [UNK] reunion tour started in 1974 i went to their [UNK] [UNK] concert and this was just as good although in [UNK] they didn't play [UNK] so i was upset they have [UNK] they have the trees they have [UNK] the pass driven [UNK] red [UNK] a [UNK] roll the bones [UNK] and much more 10 out of 10 because nothing else [UNK] if you never go to a rush concert then at least buy this dvd
    ******************************************
    It is positive with a probability of 0.961
    ground truth: 1.0
    ******************************************
    ```
    <br><br>



### Sentiment Classification Results of LSTM w/o Attention
* Loss History<br>
<img src="docs/figs/output2.jpg" width="80%"><br><br>

* Accuracy History<br>
<img src="docs/figs/output4.jpg" width="80%"><br><br>
Model w/o attention: 0.883560<br><br>

* Samples of sentiment prediction (ground truth: 1.0 (positive), 0.0 (negative))<br>
    ```
    this movie was awful plain and simple the animation scenes had absolutely terrible graphics it was very clear to see that this film had about the budget of my [UNK] bill the acting was just as bad i've seen better acting in pornographic films i would seriously like the hour and twenty minutes of my life back in fact i [UNK] on imdb just so that other people don't get sucked into watching this like i did don't get me wrong though i love scifi films this one seemed more like the intro to a video game i'm glad i only spent a dollar to see this one the story line reminded me of the movie pitch black a prisoner on a ship in outer space escapes oh my goodness what are we gonna do i would not even let this play in the background of my house while i was cleaning bottom line here you can do better
    ******************************************
    It is positive with a probability of 0.001
    ground truth: 0.0
    ******************************************


    the beloved rogue is a wonderful period piece it portrays [UNK] century paris in grand hollywood fashion yet offering a [UNK] side to existence there as it would be experienced by the poor and the snow it's constantly [UNK] about adding to the [UNK] of the setting brilliant the setting is enhanced by the odd cast of characters including [UNK] [UNK] and [UNK] a brilliant performance is turned in by john barrymore [UNK] only by the magnificent conrad [UNK] who portrays a [UNK] [UNK] louis [UNK] to perfection and yes [UNK] picks his nose on purpose pushing his portrayal to wonderfully [UNK] limits
    ******************************************
    It is positive with a probability of 0.816
    ground truth: 1.0
    ******************************************
    ```
<br><br>


### Sentiment Classification Attention Score 결과
* Positive review attention score sample<br>
<img src="docs/figs/pos.jpg" width="80%"><br><br>

* Negative review attention score sample<br>
<img src="docs/figs/neg.jpg" width="80%"><br><br>




<br><br><br>