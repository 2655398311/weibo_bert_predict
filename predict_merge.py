#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import pandas as pd
from collections import Counter
'''
预测结果处理程序
从指定的预测结果文件中读取预测结果，并统计各个分类标签出现的次数；

'''


# 读取分类
def loadLabels ():
    fn = 'labels_list.txt'
    df = pd.read_csv(fn, sep='\t', encoding='utf-8', header=None)
    df.columns=['0','1','2']
    df = df[['1','2']]
    dictR = df.set_index('2').T.to_dict('list')
    dictR= {k:''.join(v) for k,v in dictR.items()}
    #print(dictR)
    return dictR

# 读取预测结果并统计
def predict_result (filename): # encoding=encoding
    try:
        with open(filename, 'r') as f:  
            data=f.read()
        lst_ret = data.splitlines()
        # 统计时去除“其它”分类0
        lst_ret = [int(x) for x in lst_ret if not x in ['','0']]
        # 合并次数
        lstset = Counter(lst_ret)
        return lstset

    except :
        return ''
    
def main ():
    fn = './output/dat_20200306001/test_results.tsv'
    ret = predict_result(fn)
    #print(ret)
    ret_top5 = ret.most_common(5)
    #print(ret_top5)
    
    labels = loadLabels()
    #nret = {labels[k]:v for k,v in ret.items()}
    nret = [labels[k[0]] for k in ret_top5]
    print('最终提取标签Top5:')
    print(nret)


if __name__ == '__main__':
    pass
    main()
