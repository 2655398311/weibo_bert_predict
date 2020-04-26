#!/usr/bin/env sh

path='./data/Original_folder/'

echo 'Start processing unlabeled data '


cd  E://example/demo/Weibo_multi-label-classifier/
## 处理未标注的数据
sudo python weibo_dataProcess.py --path=path
##处理后的数据在 E:\example\demo\Weibo_multi-label-classifier\data\Dispose_folder 里面

'''
基于此文件夹可以采用人工标注 或 请标注团队进行标注
'''



