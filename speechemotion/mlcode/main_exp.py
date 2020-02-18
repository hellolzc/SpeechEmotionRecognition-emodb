#!/usr/bin/env python
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import time
import sklearn

from speechemotion.mlcode.pipelineCV import PipelineCV

def gen_report(result_df_stat):
    '''生成一个方便粘贴到实验报告里的表格'''
    result_report = result_df_stat.describe()
    mean = result_report.loc['mean',:]
    report = pd.DataFrame(data=[[mean['train_acc'],mean['train_precision'],mean['train_recall'],mean['train_f1score']],
                 [mean['test_acc'],mean['test_precision'],mean['test_recall'],mean['test_f1score']]], 
                 index=['train','test'], 
                 columns=['accuracy','precision','recall','F1_score'])
    return report

def save_exp_log(report_dict, name_str='', float_format='%.4f', output_dir='./log/'):
    """保存实验报告，记录实验时间和结果到文件 ./log/%Y-%m-%d_name_str.log,
    同一天同一个name_str的实验结果会追加到同一个文件中
    report_dict是一个字典，key是str，value必须可以转成str或者是DataFrame
    """
    date_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    file_name = output_dir + date_str + '_' + name_str + '.log'
    with open(file_name, mode='a') as f:
        f.write('\n' + '====' * 20 + '\n')
        f.write('Time: %s\n' % timestr)
        for key in report_dict:
            f.write('%s:' % key)
            value = report_dict[key]
            if type(value) == pd.DataFrame:
                f.write('\n')
                value.to_csv(f, sep='\t', float_format=float_format)
            else:
                f.write(str(value))
            f.write('\n')
        f.write('\n')

def main_experiment(ad_datasets, model, feature_group):
    '''实验主过程: 调用pipelineCV二十次
    '''
    seeds = list(range(1998, 2018))

    print(model)
    print('\n%s' % feature_group)

    result_df_stat = pd.DataFrame(columns=['train_acc', 'test_acc','train_precision', 'train_recall', 'train_f1score',
                                          'test_precision', 'test_recall', 'test_f1score'], dtype=float)
    pipelineCV = PipelineCV()
    pipelineCV.set_pipeline(model, ad_datasets, feature_group=feature_group)
    for indx in range(20):
        seed = seeds[indx]
        result = pipelineCV.run_pipeline(seed)
        result_df = result['cv_result']
        conf_mx = result['conf_mx']
        if indx == 0:
            conf_mx_sum = conf_mx
        else:
            conf_mx_sum += conf_mx
        for i in range(10):
            result_df_stat.loc[indx*10 + i] = result_df.loc[i]

        # show_confusion_matrix(conf_mx)
    result = {
        'result_df_stat': result_df_stat,
        'conf_mx_sum': conf_mx_sum,
        'report': gen_report(result_df_stat)
    }

    return result
