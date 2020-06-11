# Speech Emotion Recognition 

用 SVM、CNN-LSTM 进行语音情感识别。

实现了一个情感识别领域比赛常用的基线 ComParE + SVM，在Emo-DB数据集识别准确率约 88% 。

复现论文: Kim J, Saurous R A. Emotion Recognition from Human Speech Using Temporal Information and Deep Learning[C]//Interspeech. 2018: 937-940.

效果不如基线。

&nbsp;

## Environment

Python >= 3.6

&nbsp;

## Structure

```
├── speechemotion          // 所有模型的通用部分（从这里import相关的代码）
├── data                   // 存储数据和特征
├── data_ori               // 存储原始数据集
├── fusion                 // 存储处理好的特征文件，通常为一个CSV文件
├── opensmile              // Opensmile 提取特征
├── list                   // 存储实验时的数据划分，不同划分对应的结果
├── notebook               // jupyter notebook存放的位置
├── requirements.txt       // python依赖库
└── run.sh                 // 准备数据和提取特征全部流程命令
```

&nbsp;

## Requirments

### Python

```bash
pip install -r requirements.txt
```

### Tools

- [Opensmile](https://github.com/naxingyu/opensmile)：提取特征

具体参考 opensmile 目录下的 [readme.md](https://github.com/hellolzc/dementia_bank/tree/master/opensmile/scripts)


&nbsp;

## Datasets

1. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   德语，10 个人（5 名男性，5 名女性）的大约 500 个音频，表达了 7 种不同的情绪（倒数第二个字母表示情绪类别）：N = neutral，W = angry，A = fear，F = happy，T = sad，E = disgust，L = boredom。

2. [RAVDESS](https://zenodo.org/record/1188976)

   英文，24 个人（12 名男性，12 名女性）的大约 1500 个音频，表达了 8 种不同的情绪（第三位数字表示情绪类别）：01 = neutral，02 = calm，03 = happy，04 = sad，05 = angry，06 = fearful，07 = disgust，08 = surprised。支持仍待完善。

&nbsp;

## Run

1. 按`run.sh`文件中的命令运行，提取需要的特征

2. 进入`notebook`目录，运行对应的 jupyter notebook
