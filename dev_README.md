## 博客多标签提取

根据博客的内容来给博主打标签；

是对应一个还是对应多个标签: 可能对应多个标签；

爬虫逻辑：
找红人，爬取红人的博客；
还包含博客的评论的信息；

数据量：博主人数1千左右；
博客的内容，不一定；

标签：有20多个标签，这些标签是根据网上卡斯数据 对应的分类来得到的。
卡斯数据:
https://www.caasdata.com/
https://www.caasdata.com/index/rank/index.html



#评价模型：暂时没有
需要确定一个评价标准，以便于后期迭代；


##时间与预算范围

时间约定：一周；
预算范围： 1K以内；

##最终结果

能够提供以下字段
博主ID，标签词；计算的时间；


## 技术路线：
有监督的文本多标签模型；
使用ALBERT或者BERT模型训练；


参考：
https://www.jianshu.com/p/013a92bfd19e


##　所有标签：

娱乐
萌宠
时尚
音乐
街拍
新闻
直播
摄影
美食
旅行
健身
美妆
星座
生活
情感
汽车
小姐姐
小哥哥

后续有新标签会逐步更新的


https://weibo.com/u/1916401801/home?wvr=5
-----------------------------------------

博主标签：
1549362863：服装 时尚 小姐姐 直播 美妆
1669879400：明星 小姐姐 时尚 美妆
1720664360：服装 时尚 小姐姐 直播
1968758563：美妆 小哥哥 时尚
2275758460：时尚 服装 美妆 小姐姐
2970452952：美食 小姐姐
1916401801：服装 美妆 时尚 小姐姐 直播 生活 


-----------------------------------------
# 使用分类模型训练


#一: 处理标注的数据格式 (数据预处理模式)
'''
按原始数据格式存放 统一放在一个文件夹下
数据存放格式：  csv文件  里面一定要包含两列 博主ID, 博主内容
csv文件的命名如下:
xxxx.csv
```
python weibo_dataProcess.py \
	--path='加载原始文件夹路径' \
	--out='输出文件夹路径'
```
###

最终会输出处理好的csv数据
例子:
```
1549362863_sentence.csv
15999362863_sentence.csv
1765432263_sentence.csv
```

#二： 模型训练工具
提供一个模型训练工具，根据标准的训练数据，可以训练出一个分类模型。
训练数据可以不断增加，方便迭代训练，提升模型得分；
输入数据要求:

标注好的数据文件统一放在一个文件夹下面,文件后缀为csv或tsv,
数据格式为: 有标题 两列 两个字段用Tab分割
#例子: 
```
label	txt
0	我们其实不渺小，聚在一起总会成为光！
5	明晚10点，淘宝直播间公益直播  驰援一线 守护真正的天使
0	这次的疫情给我最大的感受就是无力感。
```
命令行模式 python run_classifier.py 来生成模型
```
python run_classifier.py \
	--sourcefile= /mnt/Weibo/Weibo_multi-label-classifier\data\Dispose_folder \
    --do_train=true \
	--do_eval=true \
	--do_predict=true \
	--model_name=BERT \
	--model_path=/mnt/sda1/transdat/pre_trained_model/chinese_L-12_H-768_A-12/ \
	--GPU=1 \
	--output_dir=./output/dat_20200306001/
```


#三、标签提取工具
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
	
大致思路：
'''
数据处理：
使用正则处理多标签的样本，保留第1个分类标签：
^([0-9]+),([0-9]+)\t
替换为：
\1\t

总共数据量：1200条；按8：2：10拆分；

处理情况：

```
 
正在检查数据文件: ./data/data.tsv 

None
------------数据基本情况------------
RangeIndex(start=0, stop=27769, step=1)
Index(['label', 'txt'], dtype='object')
正在检查重复数据：...
重复数据行数:0 
------------------------------
------------数据分布情况------------
数值分类个数：20
------------------------------
10    9571
0     6739
1     2282
19    2077
11    2027
16    1221
12    1031
6      746
9      464
7      353
8      336
4      292
3      226
13     147
2       91
14      84
5       32
17      26
20      20
15       4
Name: label, dtype: int64


-------------空值情况-------------
空值记录条数: 0 


 8:38:06.11|X:>filetools ./train_20200306001.csv split 7,2,1
正在拆分数据集:
数据集:./train_20200306001.csv
拆分比例:8,2,10
840  Lines data save to: ./train.tsv
240  Lines data save to: ./dev.tsv
120  Lines data save to: ./test.tsv
```
'''


把原来的分类训练代码复制过来进行修改： run_classifier.py

在win下使用ALBERT开始模拟训练：
```
python run_classifier.py --do_train=true --do_eval=true --do_predict=true --model_path=D:\model\albert_tiny_zh\ --data_dir=./data/dat_20200306001/ --output_dir=./output/dat_20200306001/
```
调整数据与代码后可以正常运行，放到GPU上使用ALBERT模型进行训练：

```
python run_classifier.py \
    --do_train=true \
	--do_eval=true \
	--do_predict=true \
	--model_path=/mnt/sda1/transdat/pre_trained_model/albert_tiny_zh/ \
	--GPU=1 \
    --data_dir=./data/dat_20200306001/ \
	--output_dir=./output/dat_20200306001/
```




-----------------------------------------
## NER模型训练

服务器目录：
/mnt/sda1/transdat/weibo_label/

修改训练程序代码，做了以下修改：
* 兼容ALBERT, 简化训练参数：
	增加model_name指定模型名称，可以是:'bert'或者'albert'（默认）
	增加model_path指定预训练模型目录； 
* 解决目录依赖问题
	复制依赖文件到当前目录，项目可以放在任意位置运行；


启动命令 for ALBERT：
训练+预测，数据集：data1
```
python run_ner.py \
    --do_train=true \
	--do_eval=true \
	--do_predict=true \
	--model_path=/mnt/sda1/transdat/pre_trained_model/albert_tiny_zh/ \
	--GPU=1 \
    --data_dir=./data/data1/
```


ALBERT for win:
```
python run_ner.py --do_train=true --model_path=X:\project\Albert_zh\prev_trained_model\albert_tiny_zh\
python run_ner.py --do_train=true --model_path=D:\model\albert_tiny_zh\
python run_ner.py --do_train=true --model_path=D:\model\albert_base\

python run_ner.py --do_train=true --model_name=bert --model_path=D:\model\chinese_L-12_H-768_A-12

```

使用albert_tiny_zh预训练模型报错：

```
ValueError: Shape of variable bert/embeddings/word_embeddings:0 ((21128, 312)) doesn't match with shape of tensor bert/e
mbeddings/word_embeddings ([21128, 128]) from checkpoint reader.
```
解决：引用ALBERT源码中的文件即可；


启动训练命令 for BERT：
```
python run_ner.py \
	--do_train=true \
	--do_eval=true \
	--do_predict=true \
	--model_name=bert \
	--model_path=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12 \
	--GPU=1 \
```

使用NER模型训练完后根本没有拟合：

```
eval_f = 0.048780486
eval_precision = 0.048780486
eval_recall = 0.048780486
global_step = 2557
loss = 342.7384

```
看来还是要回到分类，多标签的思路上；

-----------------------------------------
## 思路沟通

可西哥-NLP(xmxoxo@qq.com)  22:56:45
从微博内容给博主提取标签，比如“直播”，‘服装”，“时尚”等，一个博主可能对应多个不同的标签；

Yin_深度学习(513585416)  23:10:29
你先标好1万条微博内容对应的标签，作为训练集，
然后做个模型，训练一下，
以后有新的微博内容就可以自动标了

vkyh-NLP<yanhanwp@foxmail.com>  23:17:26
定义下相关主题词词典，然后基于Trie检索相关词。词提出来后，按照词类别和经验设定个主体标签类到样本中，然后进一步建模处理就行了

vkyh-NLP<yanhanwp@foxmail.com>  9:38:09
嗯，思路有点像2018年细粒度情感分类思路。

vkyh-NLP<yanhanwp@foxmail.com>  9:38:15
你可以查一下

vkyh-NLP<yanhanwp@foxmail.com>  9:38:35
也是20类，我当时是构建了20个分类模型搞得。。也慢

vkyh-NLP<yanhanwp@foxmail.com>  9:39:18
第一步个人觉得还是应该词典库匹配，把相应训练集构建出来

vkyh-NLP<yanhanwp@foxmail.com>  9:39:43
然后构建每个标签的分类模型，来处理哇。思路和@Yin_深度学习 说的类似

可西哥-NLP(xmxoxo@qq.com)  9:39:46
这个词典库，有点不好搞，要用什么语料呢,  人工？

vkyh-NLP<yanhanwp@foxmail.com>  9:41:37
如果空白起家的话，可能只能人工构建标签词典了。然后基于trie树在文本中找到相应核心词，出现的话就标一个这类label哇

vkyh-NLP<yanhanwp@foxmail.com>  9:42:05
标签词就看你分的范围和粒度了，能不能涵盖大部分情况。可以的话，自然词就少

可西哥-NLP(xmxoxo@qq.com)  9:37:39
比如: 
A: 时尚 服装 美妆
B：直播 生活 健身

可西哥-NLP(xmxoxo@qq.com)  9:46:34
感觉每个分类搞一个模型的话，就没办法分出权重了

可西哥-NLP(xmxoxo@qq.com)  9:48:36
比如：
A: 服装 时尚 直播 美妆
C：直播 美妆 服装 时尚
两个人内容是一样的，但是权重不同，A主要说服装与时尚；B主要是直播 美妆

Yin_深度学习(513585416)  10:01:40
不用啊，一个模型输出20个0/1

晨晓(583771987)  10:02:33
感觉 单纯靠分类这种路子 有点不太对，一个是有监督， 有多少人工 有多少智能；另外一个语义化的问题 怎么解决呢？？另外我用的 word2vec模型是基于wiki 训练的

Yin_深度学习(513585416)  10:02:39
以前做的知乎看山杯1000个标签都做了

晨晓(583771987)  10:02:59
sigmoid输出

vkyh-NLP<yanhanwp@foxmail.com>  10:03:08
Yin_深度学习  
不用啊，一个模型输出20个0/1
@Yin_深度学习 你好，请问你这个思路能说的详细点吗?

可西哥-NLP(xmxoxo@qq.com)  10:03:39
嗯，是可以输出20个分类，但多标签应该是相互有关系

可西哥-NLP(xmxoxo@qq.com)  10:03:54
20个分类权重加起来=1

Yin_深度学习(513585416)  10:04:04
最后一层是20个sigmoid

Yin_深度学习(513585416)  10:04:36
@可西哥-NLP 不要求20个分类权重加起来=1

晨晓(583771987)  10:04:45
@可西哥-NLP 多标签不是20个加起来等于1  而是 每一个都是约 0-1之间，20个加起来=1 是多分类问题

Yin_深度学习(513585416)  10:05:27
百度 多标签分类 multilabel classification

可西哥-NLP(xmxoxo@qq.com)  10:05:41
噢

晨晓(583771987)  10:06:32
分词+word2vec embedding +standard scale=》textcnn=》sigmoid 


可西哥 2020/3/5 19:47:00
大神，求多标签的BERT 代码分享

Hope -＞... 2020/3/5 19:50:59
https://github.com/huanghuidmml/cail2019_track2
-----------------------------------------
