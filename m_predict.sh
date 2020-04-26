#!/usr/bin/env sh




echo 'Start label generated '
cd  E://example/demo/Weibo_multi-label-classifier/


#file 指的是要预测的博主文件, 必须是csv格式,并且包含博主id和博主内容
# platform_cid 博主id 、 blog_content 博主内容
export file=./weibo_blog_1549362863.csv

## 使用BERT 训练模型
sudo python labelPick.py \
            --top=5
            --file=file
			--model_preDir='./model_predict/'
			----out='./output/'



