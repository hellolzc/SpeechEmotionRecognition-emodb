#!/usr/bin/env python
# python3 script
# Extract acoustic features for all audio files of the AVEC 2018 Cross-cultural Emotion Sub-Challenge (CES)
# Put the scripts into a subfolder of the AVEC2018_CES package, e.g., AVEC2018_CES/scripts/
# Output: csv files

import os
import time

# MODIFY HERE
feature_type  = 'egemaps'       # 'mfcc' or 'egemaps'
folder_data   = '/home/zhaoci/ADisease/shanghai_pictalking/data/B_speech_norm/'  # folder with audio (.wav) files
exe_opensmile = '/home/zhaoci/toolkit/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'  # MODIFY this path to the folder of the SMILExtract (version 2.3) executable
path_config   = '/home/zhaoci/toolkit/opensmile-2.3.0/config/'                                      # MODIFY this path to the config folder of opensmile 2.3 - no POSIX here on cygwin (windows)
folder_output = '../audio_features_%s/' % feature_type  # output folder
out_summary = '../func_%s.csv' % feature_type


def main():
    if feature_type=='mfcc':
        conf_smileconf = path_config + 'MFCC12_0_D_A.conf'  # MFCCs 0-12 with delta and acceleration coefficients
        opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsv 0 -timestampcsv 1 -headercsv 1'  # options from standard_data_output_lldonly.conf.inc
        outputoption = '-csvoutput'  # options from standard_data_output_lldonly.conf.inc
    elif feature_type=='egemaps':
        conf_smileconf = path_config + 'gemaps/eGeMAPSv01a.conf'  # eGeMAPS feature set
        opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'  # options from standard_data_output.conf.inc
        outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.inc
    else:
        print('Error: Feature type ' + feature_type + ' unknown!')

    if not os.path.exists(folder_output):
        os.mkdir(folder_output)
    # 删除之前的文件
    if os.path.exists(out_summary):
        os.remove(out_summary)

    for fn in os.listdir(folder_data):
        infilename  = folder_data + fn
        instname    = os.path.splitext(fn)[0]
        outfilename = folder_output + instname + '.csv'
        opensmile_call = exe_opensmile + ' ' + opensmile_options + \
                        ' -inputfile ' + infilename + ' ' + outputoption + ' ' + outfilename + \
                        ' -instname ' + instname + ' -output ?' + ' -csvoutput '+ out_summary  # (disabling htk output)
        os.system(opensmile_call)
        time.sleep(0.01)

    os.remove('smile.log')


if __name__ == '__main__':
    main()
