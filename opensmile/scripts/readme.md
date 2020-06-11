# 声学特征提取

## 环境配置

* Python>=3.6
* Python 包: numpy, pandas
* openSMILE https://www.audeering.com/opensmile/

openSMILE下载完成后要修改`extract_audio_features.py`和`extract_is09etc_features.py`的
`exe_opensmile` 和 `path_config` 的值，使其指向openSMILE可执行文件的位置和特征配置文件的位置

建议将脚本都放在一个scripts的目录下, 生成的特征文件会放在和scripts同一层的目录中
openSMILE工具调用的时候有点问题，请尽量使用绝对路径，不要使用相对路径，而且文件名开头不要是中文。


## 支持特征

- `mfcc`: 39维的时序特征；
- `eGeMAPs`: [论文：eGeMAPS特征集（2016 IEEE trans on Affective Computing）](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7160715), 88个特征;
- `IS09_emotion`：[The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf)，384 个特征；
- `IS10_paraling`：[The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf)，1582 个特征；
- `IS11_speaker_state`：[The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf)，4368 个特征；
- `IS12_speaker_trait`：[The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf)，6125 个特征；
- `IS13_ComParE`：[The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf)，6373 个特征；
- `ComParE_2016`：[The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf)，6373 个特征。


## 脚本说明

共6个脚本文件

| 文件名                        | 描述                                               |
|     ---                       | ---                                                |
| extract_all.sh                | 调用extract_is09etc_features.py和convert_stat_format.py两个脚本   |
| extract_audio_features.py     | 用来提取mfcc和egemaps特征, 建议使用下面的脚本替代     |
| extract_is09etc_features.py   | 用来提取上面提到的特征，支持命令行参数   |
| convert_stat_format.py        | 将提取出的统计特征转成标准的CSV格式，支持命令行参数     |
| feature_viewer.ipynb          | 用来查看提取的时序特征的内容     |
