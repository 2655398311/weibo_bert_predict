# weibo_bert_predict
weibo_bert_predict
微博多标签分类

根据微博博主的近半年的微博，对每个博主进行打标签
##技术实现方案：
有监督的文本多标签分类
使用ALBERT或者BERT模型进行训练

当前有24类微博标签  此标签存放在labels.txt文本中
recreation_Film	娱乐影视	1
cute	萌宠	2
fashion	时尚	3
music	音乐	4
street	街拍	5
news	新闻	6
live	直播	7
photography	摄影	8
cate	美食	9
travel	旅行	10
fitness	健身	11
beauty	美妆	12
sign 	星座	13
life_Wiki	生活百科	14
girl	小姐姐	15
emotion	情感	16
car	汽车	17
brother	小哥哥	18
clothes	服装	19
star	明星	20
Funny_humor                搞笑幽默    21
morther_baby      母婴    22
Art_Attack_Design  艺术创意设计   23
Knowledge_Education         知识文化               24
other	其它	0


##一：数据预处理模块
把拿到的原始微博数据处理成待标注的数据格式（数据的预处理模式）
'''
  python weibo_dataProcess.py \
	--path='././data/Original_folder/
'''
参数说明
path:
按原始数据格式存放，统一放在一个文件中
数据存放的格式必须为csv格式，数据格式必须包含两列  
columns1(platform_cid)博主的微博ID    columns2(blog_content)对应的博主的微博内容
运行此代码输出结果
最终会在data目录下生成Dispose_folder文件夹 作为处理过待标注的csv数据
预处理后待标注的csv数据格式例如下：
'''
label	txt
0	这个动作一定要注意肩胛骨内收，先内收肩胛骨，再回拉手臂，顶点的时候想象背部夹住了一根绳子，实现念动一致孤立发力。
0	天才小画家们一起出慈善联名款
0	以及神秘却巨诱惑力的探险王国
0	超有意思，快来康康
0	决定明年开始不孤军奋战了，
0	杀人回忆凶手原型被抓
0	专门用来感谢亲爱滴铁粉小伙伴们，以及看着眼熟的粉丝（铁粉标识掉了的小伙伴莫慌，这主要赖我最近发博不够频繁）。
0	分享越多，朋友下单越多，你赚得越多！
'''

-----------------------------------------------------------------------------------------
模型训练工具
简述：
提供一个模型的训练工具,传入标准的训练数据，可以训练出一个分类模型
后期训练数据不断增加，进行迭代训练 提升模型的各项得分

使用命令 '''python3 run_classifier.py'''  来训练模型
训练模型需传入参数  
```
python run_classifier.py \
	--sourcefile= ./data/Dispose_folder \
        --do_train=true \
	--do_eval=true \
	--do_predict=false \
	--model_name=BERT \
	--model_path=./chinese_L-12_H-768_A-12/ \
	--output_dir=./model_train/
```
本人设置默认采用GPU进行模型训练  增加模型整体的训练速度与效率

参数说明：
sourfile:
标注好的数据文件统一放在一个文件夹下面，文件后缀为csv或者是tsv
数据格式为 ：有标题 两个字段用tab分割
do_train:是否在训练集上训练模型
do_eval:是否在验证集上验证模型
do_predict:是否在测试集上测试模型
model_name:模型  ALBERT或者是BERT
model_path:如果为BERT模型的话  加载的模型路径为 chinese_L-12_H-768_A-12,如果是Albert模型 需要加载的预训练模型为:albert_tiny_zh
output_dir:为最后模型的输出路径 这里输出的路径为  model_train


训练数据sourfile的数据格式如下：
```
label	txt
10	南方落雪
10	带着微博去旅行全球旅行攻略最炫打卡地【厦门自由行攻略】逛吃，逛吃，不避开这些坑的话逛不好、吃不爽
1	Keanna是中荷混血，做过模特，一直是谢和弦的忠实粉丝，陪他戒毒、治病，后来嫁给了他。
19	秋天第一波A字米色风衣 大姨妈出品的风衣口碑大家也都知道哦 洗水和版型是我最在意和考究的 包括里布的用料…
10	快闪，来吧小伙伴们，稻城亚丁欢迎你！稻城亚丁旅游攻略西藏旅游攻略 稻城亚丁小胡胡的秒拍视频
10	全球旅行攻略遇见美好 大西洋西岸巴哈马的沙滩，海，和动物们
10	全球旅行攻略不可辜负的美食【越南旅游攻略】人均000元的自由行一份街头河粉 一季越南夏天享受夏季的炙热与温柔久违法式风情的感动步步寻忘迹，有处特依依。
10	九寨沟旅游攻略看水必去九寨沟 10月有约的吗？
```
训练模型评价结果：

```
eval_accuracy = 0.850018
eval_f1 = 0.9229107
eval_loss = 0.71012884
eval_precision = 0.9198181
eval_recall = 0.9260241
global_step = 8330
loss = 0.7093687
```


##三：标签提取  （模型预测）
简述:
标签提取工具是一个命令行工具，按照要求的格式把每个博主的所有博文保存在一个文件中，
使用命令行可以对单个文件或者目录下的多个csv文件批量处理，提取出各个博主对应的热门标签；

标签提取命令行工具规划：labelPick.py
参数规划：
```
--top N 提取N个最热门标签
--path 文件夹   加载博主的数据路径,必须为csv格式
--out 文件名  输出标签提取的文件名路径
--model_preDir 此处加载训练好的模型(pb或ckpt)路径
```
使用样例：
```
python labelPick.py \
            --top=5 \
            --path=./blog_floder/ \
			--model_preDir=./model_predict/ \
			--out=./output/
```

参数说明:

```
--top N 提取N个最热门标签
--path 文件夹 里面的文件数据格式: 1. 必须包含  platform_cid(博主ID), blog_content(博主内容) 这两个字段 2.文件的后缀必须是以csv结尾的
--model_preDir : 此处存放训练好的pb或ckpt模型, 将得分最好的模型文件放到此文件夹下
--out 输出最终结果 样例下方有说明
```
###注意: model_preDir 此处放的时得分最好的ckpt或pb模型, 目前支持的是ckpt模型的批量预测, 将得分最好的ckpt模型放到此文件夹下
输出文件格式为：有标题行，一行一条记录,，用TAB分隔
输出文件样例如下：

```
blog_ID	Top-all	Top-k	time
博主ID	全部标签	服装,音乐,直播,美食,美妆	1585721976
博主ID	全部标签	明星,时尚,美妆,旅游,摄影	1585721976
```
###sh命令文件
```
我将这三个工具的命令都封装到 sh文件里面了, 以后可以直接运行sh文件
分别是:
m_process.sh :数据预处理工具
m_train_Albert.sh： 使用ALBert 模型训练工具
m_train_bert.sh: 使用Bert 模型训练工具
m_predict.sh: 标签提取工具
```

##模型我最后一次训练的数据情况

在验证集上的模型得分:
eval_accuracy = 0.850018
eval_f1 = 0.9229107
eval_loss = 0.71012884
