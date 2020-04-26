#!/usr/bin/env sh

echo 'Start train  data '
cd  E://example/demo/Weibo_multi-label-classifier/


#sourcefile 指的是标注好的数据路径
export sourcefile='./data/Dispose_folder/'
export AlBERT_BASE_DIR='./albert_tiny_zh/'
export output_dir='./model_train/'


## 使用ALBERT 训练模型
sudo python run_classifier.py \
            --sourcefile=sourcefile \
            --do_train=true \
            --do_eval=true \
            --do_predict=false \
            --model_name=ALBERT \
            --model_path=AlBERT_BASE_DIR \
            --GPU=0 \
            --output_dir=output_dir