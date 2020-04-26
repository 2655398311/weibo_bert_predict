# -*- coding: utf-8 -*-
'''
@author: frf@sina.com
@license: (C) Copyright 2019
@desc: 项目执行参数
@DateTime: Created on 2019/7/26, at 下午 02:04 by PyCharm
'''
import os
import datetime
now = datetime.datetime.now()
BERT_BASE_DIR = os.path.join('./','chinese_L-12_H-768_A-12/')
arg_dic = {
    "data_dir": os.path.join(os.getcwd(),'data/dataAll'),              # 数据目录
    "Dispose_folder":os.path.join(os.getcwd(),'data/Dispose_folder'),
    "output_predict":os.path.join(os.getcwd(),'output/') ,
    'eval_dir': os.path.join(os.getcwd(),'eval_information/'),
    "bert_config_file": os.path.join(BERT_BASE_DIR,'bert_config.json'),
    "task_name": 'cnews',  # "The name of the task to train.
    "vocab_file": os.path.join(BERT_BASE_DIR,'vocab.txt') ,  # The vocabulary file that the BERT model was trained on.
    "init_checkpoint": os.path.join(BERT_BASE_DIR,'bert_model.ckpt'),
    # "Initial checkpoint (usually from a pre-trained BERT model).
    "do_lower_case": True,
    "id": 0,
    "max_seq_length": 150,
    "do_train": False,
    "do_eval": False,
    "do_predict": True,
    "train_batch_size": 6,
    "eval_batch_size": 8,
    "predict_batch_size": 8,
    "learning_rate": 3e-5,
    "num_train_epochs": 5,
    "warmup_proportion": 0.1,  # "Proportion of training to perform linear learning rate warmup for. "
    # "E.g., 0.1 = 10% of training."
    "save_checkpoints_steps": 3000,  # How often to save the model checkpoint."
    "iterations_per_loop": 1000,  # "How many steps to make in each estimator call.
    "use_tpu": False,
    "tpu_name": False,
    "tpu_zone": False,
    "gcp_project": False,
    "master": False,
    "num_tpu_cores": False,  # "Only used if `use_tpu` is True. Total number of TPU cores to use."
}
