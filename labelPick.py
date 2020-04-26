#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
标签提取工具是一个命令行工具，按照要求的格式把每个博主的所有博文保存在一个文件中，
使用命令行可以对单个文件或者目录下的多个txt文件批量处理，提取出各个博主对应的热门标签；

标签提取命令行工具规划：labelPick.py
参数规划：
```
--top N 提取N个最热门标签
--file 文件名   单个数据文件处理
--path 目录名	自动读取.txt文件
--out 目录名	指定输出结果文件的目录，文件名为labels.txt,默认目录为当前目录，会自动覆盖旧文件
```
使用样例：
```
labelPick.py --top 5 --file ./weibo.txt --out ./out/
labelPick.py --path ./dat/ --out ./out/
```
输出文件为：有标题行两列（文件名，热门标签），一行一条记录,，用TAB分隔
输出文件样例：
```
filename	labels
111.txt	服装,音乐,直播,美食,美妆
222.txt	明星,时尚,美妆,旅游,摄影
```

'''
import argparse
import numpy as np
import json
import logging
import os
import pandas as pd
import re
import sys
import traceback
from weibo_dataProcess import data_analye
from predict_GPU import Bert_Class,arg_dic
from collections import Counter
import  csv
import time
from sklearn.metrics import classification_report
from sklearn import metrics


# get all files and floders in a path
# fileExt: ['png','jpg','jpeg']
# return: 
#    return a list ,include floders and files , like [['./aa'],['./aa/abc.txt']]
def getFiles (workpath, fileExt = []):
    try:
        lstFiles = []
        lstFloders = []

        if os.path.isdir(workpath):
            for dirname in os.listdir(workpath) :
                #file_path = os.path.join(workpath, dirname)
                file_path = workpath  + '/' + dirname
                if os.path.isfile( file_path ):
                    if fileExt:
                        if dirname[dirname.rfind('.')+1:] in fileExt:
                           lstFiles.append (file_path)
                    else:
                        lstFiles.append (file_path)
                if os.path.isdir( file_path ):
                    lstFloders.append (file_path)      

        elif os.path.isfile(workpath):
            lstFiles.append(workpath)
        else:
            return None
        
        lstRet = [lstFloders,lstFiles]
        return lstRet
    except Exception as e :
        return None


#混淆矩阵的打印
def skl_getMatrix():
    pass
    path = r'D:\work_space\Weibo\Weibo_multi-label-classifier\data\dataAll\dat_20200403'
    model_preDir = r'./model_predict/'
    filename = os.path.join(arg_dic['output_predict'], 'test_results.tsv')
    dev = os.path.join(path,'dev.tsv')
    y_true = []
    lines = []
    with open(dev,'r',encoding='utf-8') as f:
        reader = csv.reader(f,delimiter="\t")
        for line in reader:
            y_true.append(int(line[0]))
            lines.append(line[1])
    bc = Bert_Class(model_preDir)
    bc.predict_on_ckpt(lines)
    with open(filename, 'r') as f:
        data = f.read()
    lst_ret = data.splitlines()
    y_pred = [int(x) for x in lst_ret]
    print(y_true)
    print(y_pred)
    eval_report = classification_report(y_true,y_pred)
    print(eval_report)
    return  0


def filter_text(text):
    new_text = text
    new_text = re.sub(r'<br>',r'\n',new_text)
    new_text = re.sub(r'抱歉，此微博已被作者删除。查看帮助：',r'',new_text)
    new_text = re.sub(r'抱歉，由于作者设置，你暂时没有这条微博的查看权限哦。查看帮助：',r'',new_text)
    new_text = re.sub(r'分享图片',r'',new_text)
    re_tag = re.compile(\
        '</?\w+[^>]*>|'
        '<img src=(.*)?>|'
        '(<img)? src=([^>]*)?>|'
        '<(.*)>|'
        '抱(.*)>|'
        '\[.*?\]|'
        '\【.*?\】|'
        "http(.*)|"
        "(&gt;|nan|,+)+"
        ,re.I)
    new_text = re.sub(re_tag,'',new_text)
    new_text = re.sub("[～#┌―┐└┘┐~╭(╯3╰)╮]", "", new_text)
    #new_text = re.sub("[.+|…|。+|\n|\t|—+| |の|→_→]+", "。", new_text)  # 合并句号
    new_text = re.sub(r'([。？！…?!]+(?:”|"*))', r'\1\n', new_text) #分句
    new_text = re.sub('(\n\s+)',r"\n",new_text)  # blank line

    if new_text:
        if new_text[-1]!='\n':
          new_text += '\n'
          pass
    # print(new_text)
    return  new_text

# 预处理，过滤，分句；
# 输入： 文本
# 输出： 处理好的句子list
def preprocess (path):
    pass
    print('正在读取微博数据...')
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    data = df.dropna(subset=['blog_content'])
    weibo_id = df['platform_cid'][1]
    data = np.array(data['blog_content'])
    data_list = list(filter(None, data.tolist()))  # 只能过滤空字符和None
    result = []
    for i in range(len(data_list)):
        tx = data_list[i]
        if tx:
            text = filter_text(tx)
            for x in text.splitlines():
                x = x.strip()
                if len(x) > 3:
                    result.append(x)
    print(len(result))
    return  result,weibo_id
# 数据预测，对句子进行预测，得到分类结果；
# 输入：句子的list
# 输出：预测分类号的list
def predict (lstTxt,model_preDir):
    pass
    res_pre = []
    filename = os.path.join(arg_dic['output_predict'], 'test_results.tsv')
    bc = Bert_Class(model_preDir)
    for i in os.listdir(model_preDir):
        if '.pb' in i:
            for j in lstTxt:
                res_pb = bc.predict_on_pb(j)
                res_pre.append(res_pb)
            with open(filename,'w',newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in res_pre:
                    writer.writerow(row)
        elif 'checkpoint' in i:
            bc.predict_on_ckpt(lstTxt)
    time_end = int(time.time())
    result = []
    result.append(time_end)
    try:
        with open(filename, 'r') as f:
            data = f.read()
        lst_ret = data.splitlines()
        lst_ret1 = [int(x) for x in lst_ret]
        result.append(lst_ret1)
        # 统计时去除“其它”分类0
        lst_ret = [int(x) for x in lst_ret if not x in ['', '0']]
        # 合并次数
        lstset = Counter(lst_ret)
        result.append(lstset)
        return result

    except:
        return ''


# 从分类文件中加载分类名称
# 输入：文件名, 
# 输出：分类号与名称的对应字典，{0:'其它',...}
def load_labels (fn):
    pass
    fn = 'labels_list.txt'
    df = pd.read_csv(fn, sep='\t', encoding='utf-8', header=None)
    df.columns=['0','1','2']
    df = df[['1','2']]
    dictR = df.set_index('2').T.to_dict('list')
    dictR= {k:''.join(v) for k,v in dictR.items()}
    #print(dictR)
    return dictR


# 分类结果统计，提取热门标签
# 输入: 分类号list； 提取的个数top,默认=5
# 输出：热门标签字符串
def top_label (lstClass,bozhu_id,top,out):
    pass
    if  not os.path.exists(out):
        os.makedirs(out)
    dict_all = {}
    dict_all['blog_ID']= bozhu_id
    fn = './labels_list.txt'
    labels = load_labels(fn)
    print(labels)
    ret_top = lstClass[-1].most_common(int(top))

    count_all = Counter(lstClass[1])
    ret_top_all = count_all.most_common(20)
    d = {i[0]: '%.4f' % (i[1] / len(lstClass[1])) for i in ret_top_all}
    label_ratio = sorted(d.items(),key=lambda item:item[1] ,reverse=True)
    #nret = {labels[k]:v for k,v in ret.items()}
    print('从高到地对类别数进行排序:')
    print(label_ratio)
    nret1 = [labels[k[0]] for k in ret_top]
    nret = [(labels[k[0]],float(k[1])) for k in label_ratio]
    print('最终提取标签Top-{}:'.format(int(top)))
    print(nret1)
    # dict_all['Top-all'] = nret
    dict_all['Top-{}'.format(int(top))] = nret1
    dict_all['time'] = lstClass[0]
    print(dict_all)
    df = pd.DataFrame([dict_all])
    df.to_csv(os.path.join(out,'result.csv'),encoding='utf-8',index=None)
    return 0


# 命令行处理
def main():
    parser = argparse.ArgumentParser(description='微博标签提取工具')
    parser.add_argument('--top', default=5, type=int,  help='提取最热门标签的个数，默认=5')
    parser.add_argument('--path', default='./blog_floder/', help='批量处理的数据目录')
    parser.add_argument('--model_preDir', default='./model_predict/', help='此处加载训练好的模型(pb或ckpt)')
    parser.add_argument('--out', default='./output/', required=False, help='输出目录')

    args = parser.parse_args()
    path = args.path
    top = args.top
    out = args.out
    model_preDir = args.model_preDir
    for dirname in os.listdir(path):
        if dirname.split('.')[-1] == 'csv':
            file_path = os.path.join(path, dirname)
            print(file_path)
            result,bozhu_id = preprocess(file_path)
            result_li = predict(result,model_preDir)
            top_label(result_li,bozhu_id,top,out)



if __name__ == '__main__':
    pass
    main()
    #获取在验证集上的混淆矩阵
    # skl_getMatrix()