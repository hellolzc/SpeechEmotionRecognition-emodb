# 声学特征提取

## 环境配置

* Python>=3.6
* Python 包: numpy, pandas
* openSMILE https://www.audeering.com/opensmile/

openSMILE下载完成后要修改`extract_audio_features.py`和`extract_is09etc_features.py`的
`exe_opensmile` 和 `path_config` 的值，使其指向openSMILE可执行文件的位置和特征配置文件的位置

建议将脚本都放在一个scripts的目录下, 生成的特征文件会放在和scripts同一层的目录中
openSMILE工具调用的时候有点问题，请尽量使用绝对路径，不要使用相对路径，而且文件名开头不要是中文。

## 脚本说明

共6个脚本文件

| 文件名                        | 描述                                               |
|     ---                       | ---                                                |
| convert_stat_format.py        | 将提取出的统计特征转成标准的CSV格式，支持命令行参数     |
| extract_all.sh                | 调用extract_is09etc_features.py和convert_stat_format.py两个脚本   |
| extract_audio_features.py     | 用来提取mfcc和egemaps特征     |
| extract_is09etc_features.py   | 用来提取InterSpeech 09/10/11/12/13、ComParE特征，也可以提取eGeMAPs特征，支持命令行参数   |
| feature_viewer.ipynb          | 用来查看提取的时序特征的内容     |
