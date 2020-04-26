# -*- coding: utf-8 -*-
# @Time    : 2020/2/28 17:34
# @Author  : FRF
# @File    : weibo_dataProcess.py
# @Software: PyCharm
'''
数据预处理工具
'''
import os
import re
import numpy as np
import pandas as pd
import csv
import argparse
import time
from arguments import  arg_dic

def split_data(x):
    # x1 = int(x)
    x = str(x).replace('、','0')
    x = str(x).replace('nan','0')
    x = str(x).replace('，',',')
    x = str(x).replace('0.0','0')
    x = str(x).replace('1.0','1')
    if len(str(x))>1:
        x = str(x).split(',')[0]
    if x != 'nan' and int(x)>20:
        x = 0
    return  str(x)

class data_analye(object):

    ## 使用pandas对样本数据文件进行检查；
    # drop_dup 是否删除重复数据
    @classmethod
    def pd_datCheck(cls,lstFile, drop_dup=0, header=None):
        pass
        try:
            print("正在检查数据文件: %s \n" % lstFile)
            print(header)
            df = pd.read_csv(lstFile, delimiter="\t")
            print("数据基本情况".center(30, '-'))
            print(df.index)
            print(df.columns)
            # print(df.head())
            print('正在检查重复数据：...')
            dfrep = df[df.duplicated()]
            print('重复数据行数:%d ' % len(dfrep))
            if len(dfrep) > 0:
                print(dfrep)
            if drop_dup and len(dfrep):
                print('正在删除重复数据：...')
                df = df.drop_duplicates()
                df.to_csv(lstFile, index=0, sep='\t')
            print('-' * 30)
            print("数据分布情况".center(30, '-'))
            dfc = df[df.columns[0]].value_counts()
            print('数值分类个数：%d' % len(dfc))
            print('-' * 30)
            print(dfc)
            print('\n')
            print("空值情况".center(30, '-'))
            df.dropna(axis=0, how='any', inplace=True)
            dfn = df[df.isnull().values == True]
            print('空值记录条数: %d ' % len(dfn))
            if len(dfn) > 0:
                print('空记录'.center(30, '-'))
                print(dfn.head())
            print('\n')
            return 0
        except Exception as e:
            print("Error in pd_dat:")
            print(e)
            return -1

    # 删除0标签的数据
    @classmethod
    def _read_csv(cls):
        f = open('./data/data2.tsv', encoding='utf-8', mode='r+')
        L = list(csv.reader(f))
        aa = []
        for i in range(0, len(L)):
            label = L[i][0].split('\t')[0]
            txt = L[i][0].split('\t')[1]
            if label != '0':
                aa.append({'label': label, 'txt': txt})
        print(aa)
        data = pd.DataFrame(aa)
        print(data.head())
        # data = data.T
        # data.rename(columns={0:'label',1:'txt'},inplace=True)
        data.to_csv('./aa.tsv', encoding='utf-8', sep='\t', index=False)

    # 处理数据集数据
    @classmethod
    def dataPro(cls,sourcefile='./data/', outpath=''):
        pass
        if not os.path.exists(outpath):
            print('目录%s不存在，请检查!' % outpath)
            os.makedirs(outpath)
        for path in os.listdir(sourcefile):
            if path.split('.')[-1] == 'csv' or 'tsv':
                dataAll = os.path.join(sourcefile, path)
                data = pd.read_csv(dataAll, sep='\t', encoding='utf-8')
                # data.columns = ['label', 'txt']
                data1 = pd.DataFrame(data)
                data1['label'] = data['label'].apply(lambda x: split_data(x))
                output = os.path.join(outpath, 'dataAll.tsv')
                data1.to_csv(output, mode='a', sep='\t', encoding='utf-8', index=False,
                             header=False)
        return output

    '''
    过滤html标签, 超链接,图片等
    :return:
    '''
    @classmethod
    def filter_text(cls,text):
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
        #new_text = re.sub(re.compile(r'^(.{1,3})\n',re.M), r'', new_text) # 去除太短的行
        #new_text = re.sub(r'\n(\s*)+', r'\n', new_text)
        new_text = re.sub('(\n\s+)',r"\n",new_text)  # blank line

        if new_text:
            if new_text[-1]!='\n':
              new_text += '\n'
              pass
        # print(new_text)
        return  new_text

    @classmethod
    def batch_process(cls,path):
        pass
        #读取微博数据
        print('正在读取微博数据...')
        #df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        df = pd.read_csv(path, sep=',', encoding='utf-8')
        data = df.dropna(subset=['blog_content'])
        data = np.array(data['blog_content'])
        weibo_id = df['platform_cid'][1]
        print(weibo_id)
        data_list = list(filter(None, data.tolist()))  # 只能过滤空字符和None
        print(len(data_list))
        maxlen = 0
        maxtxt = ''
        maxline = 0
        maxsrc = ''
        sret = ''
        for i in range(len(data_list)):
            if i==10:
                pass
                #return
            #print('-'*30)
            tx = data_list[i]
            if tx:
                #print(tx)
                text = cls.filter_text(tx)
                st = ''
                for x in text.splitlines():
                    x = x.strip()
                    if len(x)>maxlen:
                        maxlen = len(x)
                        maxtxt = x
                        maxline = i
                        maxsrc = tx
                    if len(x)>3:
                        st += '0\t'+ x + '\n'
                #print('-----result-----')
                #print(text)
                if st:
                    sret += st #+ '\n'
        print(arg_dic['Dispose_folder'])
        print(weibo_id)
        outfile = os.path.join(arg_dic['Dispose_folder'],str(weibo_id) +'_sentence.csv')
        with open(outfile , 'w', encoding='utf-8') as f:
            f.write(sret)  #'label\ttxt\n'+
        print('max line: %d\n%s' % (maxline,maxsrc))
        print('-'*30)
        print('max line length: %d\n%s' % (maxlen,maxtxt))


    # 切分csv格式的数据集
    @classmethod
    def CreateCategoryTrain(cls,path='./data/data.tsv', rebuild=0):
        pass
        fnout = os.path.join(path, 'dat_'+ time.strftime('%Y%m%d'))
        if rebuild:
            print('指定强制重新生成标注文件...')
        else:
            if not os.path.exists(fnout):
                os.makedirs(fnout)
                print('训练文件已生成,跳过生成步骤...')
        dfn = pd.read_csv(os.path.join(path,'dataAll.tsv'), encoding='utf-8', sep='\t')
        # print(dfn.head())
        # 删除完全相同的数据行
        dfn = dfn.drop_duplicates()  # keep='first', inplace=True)
        # 删除含有空值的行
        dfn = dfn.dropna(axis=0, how='any')
        # 数据随机打乱
        dfn = dfn.sample(frac=1)

        # 数据集切分: train:dev:test = 8:2:10
        intTotal = dfn.shape[0]
        intCut = int(intTotal * 0.8)
        df_train = dfn.head(intCut)
        df_dev = dfn.tail(intTotal - intCut)

        # 保存数据  #,sep='\t'
        df_train.to_csv(os.path.join(fnout, 'train.tsv'), index=False, sep='\t')
        df_dev.to_csv(os.path.join(fnout, 'dev.tsv'), index=False, sep='\t')
        dfn.to_csv(os.path.join(fnout, 'test.tsv'), index=False, sep='\t')
        print('分类模型训练数据已生成。')
        return fnout

# 命令行处理
def main():
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--path', default='./data/Original_folder/', help='批量处理的数据目录')
    # parser.add_argument('--out', default='./', required=False, help='输出目录')
    args = parser.parse_args()
    path = args.path
    for dirname in os.listdir(path):
        if dirname.split('.')[-1] == 'csv':
            file_path = os.path.join(path, dirname)
            print(file_path)
            data_analye.batch_process(file_path)




if __name__ == '__main__':
    pass
    main()




