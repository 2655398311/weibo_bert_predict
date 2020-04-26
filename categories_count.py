# -*- coding: utf-8 -*-
# @Time : 2020/3/13 20:40
# @Author : frf
# @File : categories_count.py
# @Software: PyCharm

import  pandas as pd
import csv
## 使用pandas对样本数据文件进行检查；
# drop_dup 是否删除重复数据
def pd_datCheck (lstFile, drop_dup=0, header=None):
    pass
    try:
        print("正在检查数据文件: %s \n" % lstFile)
        print(header)
        df = pd.read_csv(lstFile, delimiter="\t")
        print("数据基本情况".center(30,'-'))
        print(df.index)
        print(df.columns)
        #print(df.head())
        print('正在检查重复数据：...')
        dfrep = df[df.duplicated()]
        print('重复数据行数:%d ' % len(dfrep))
        if len(dfrep)>0:
            print(dfrep)
        if drop_dup and len(dfrep) :
            print('正在删除重复数据：...')
            df = df.drop_duplicates()
            df.to_csv(lstFile, index=0, sep = '\t')
        print('-'*30)
        print("数据分布情况".center(30,'-'))
        dfc = df[df.columns[0]].value_counts()
        print('数值分类个数：%d' % len(dfc))
        print('-'*30)
        print(dfc)
        print('\n')
        print("空值情况".center(30,'-'))
        df.dropna(axis=0, how='any', inplace=True)
        dfn = df[df.isnull().values==True]
        print('空值记录条数: %d ' % len(dfn))
        if len(dfn)>0:
            print('空记录'.center(30,'-'))
            print(dfn.head())
        print('\n')
        return 0
    except Exception as e :
        print("Error in pd_dat:")
        print(e)
        return -1

def _read_csv():
    f = open('./data/data2.tsv',encoding='utf-8',mode='r+')
    L = list(csv.reader(f))
    aa = []
    for i in range(0,len(L)):
        label = L[i][0].split('\t')[0]
        txt = L[i][0].split('\t')[1]
        if label != '0':
            aa.append({'label':label,'txt':txt})
    print(aa)
    data = pd.DataFrame(aa)
    print(data.head())
    # data = data.T
    # data.rename(columns={0:'label',1:'txt'},inplace=True)
    data.to_csv('./aa.tsv',encoding='utf-8',sep='\t',index=False)



if __name__ == '__main__':
    pass
    path = r'./data/data.tsv'
    pd_datCheck(path,drop_dup=1)
    # _read_csv()