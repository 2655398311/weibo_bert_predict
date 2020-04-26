# -*- coding: utf-8 -*-
'''
@author: yaleimeng@sina.com
@license: (C) Copyright 2019
@desc: 这个代码是进行预测的。既可以根据ckpt检查点，也可以根据单个pb模型。
@DateTime: Created on 2019/7/19, at 下午 04:13 by PyCharm
'''


from train_eval import  *
from arguments import  arg_dic
import tensorflow as tf
from tensorflow.python.estimator.model_fn import EstimatorSpec
import numpy as np

class Bert_Class():

    def __init__(self,model_preDir):
        self.graph_path = os.path.join(model_preDir, 'classification_model.pb')
        if os.path.exists(self.graph_path):
            self.runfunc = self.load_graph(self.graph_path)
        self.ckpt_tool, self.pbTool = None, None
        self.prepare(model_preDir)


    def load_graph(self,model_path):

        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print("load model...")
            return graph_def


    def classification_model_fn(self, features, mode,):

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_map = {"input_ids": input_ids, "input_mask": input_mask}
        pred_probs = tf.import_graph_def(self.runfunc, name='', input_map=input_map, return_elements=['pred_prob:0'])

        return EstimatorSpec(mode=mode, predictions={
            'encodes': tf.argmax(pred_probs[0], axis=-1),
            'score': tf.reduce_max(pred_probs[0], axis=-1)})

    def prepare(self,model_preDir):
        print(model_preDir)
        tokenization.validate_case_matches_checkpoint(arg_dic['do_lower_case'], arg_dic['init_checkpoint'])
        self.config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])

        if arg_dic['max_seq_length'] > self.config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (arg_dic['max_seq_length'], self.config.max_position_embeddings))
        if not os.path.exists(model_preDir):
            tf.gfile.MakeDirs(model_preDir)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=arg_dic['vocab_file'],
                                                    do_lower_case=arg_dic['do_lower_case'])

        self.processor = SelfProcessor()
        # self.train_examples = self.processor.get_train_examples(arg_dic['data_dir'])
        global label_list
        label_list = self.processor.get_labels()

        self.run_config = tf.estimator.RunConfig(
            model_dir=model_preDir, save_checkpoints_steps=arg_dic['save_checkpoints_steps'],
            tf_random_seed=None, save_summary_steps=100, session_config=None, keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=10000, log_step_count_steps=100, )

    def predict_on_ckpt(self,exam):
        if not self.ckpt_tool:
            num_train_steps = int(len(exam) / arg_dic['train_batch_size'] * arg_dic['num_train_epochs'])
            num_warmup_steps = int(num_train_steps * arg_dic['warmup_proportion'])

            model_fn = model_fn_builder(bert_config=self.config, num_labels=len(label_list),
                                        init_checkpoint=arg_dic['init_checkpoint'], learning_rate=arg_dic['learning_rate'],
                                        num_train=num_train_steps, num_warmup=num_warmup_steps)

            self.ckpt_tool = tf.estimator.Estimator(model_fn=model_fn, config=self.run_config, )
        # exam = self.processor.one_example(sentence)  # 待预测的样本列表
        examples = self.processor._create_examples(exam,'test')
        num_actual_predict_examples = len(examples)
        predict_file = os.path.join(arg_dic['output_predict'], "predict.tf_record")
        file_based_convert_examples_to_features(examples, label_list,
                                                arg_dic['max_seq_length'], self.tokenizer,
                                                predict_file)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(examples), num_actual_predict_examples,
                        len(examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", arg_dic['predict_batch_size'])

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                            seq_length=arg_dic['max_seq_length'], is_training=False,
                                            drop_remainder=False)

        result = self.ckpt_tool.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        output_predict_file = os.path.join(arg_dic['output_predict'], "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= len(exam):
                    break
                output_line = str(np.argmax(probabilities)) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == len(exam)
        # gailv = list(result)[0]["probabilities"].tolist()
        # pos = gailv.index(max(gailv))  # 定位到最大概率值索引，
        # return label_list[pos]

    def predict_on_pb(self, sentence):
        if not self.pbTool:
            self.pbTool = tf.estimator.Estimator(model_fn=self.classification_model_fn, config=self.run_config, )
        examples = self.processor.one_example(sentence)  # 待预测的样本列表
        feature = convert_single_example(0, examples, label_list, arg_dic['max_seq_length'], self.tokenizer)
        predict_input_fn = input_fn_builder(features=[feature, ],
                                            seq_length=arg_dic['max_seq_length'], is_training=False,
                                            drop_remainder=False)
        result = self.pbTool.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        print(result)
        ele = list(result)
        print(ele)
        label = label_list[ele['encodes']]
        print('类别：{}，置信度：{:.3f}'.format(label_list[ele['encodes']], ele['score']))
        return label


if __name__ == '__main__':
    pass
    model_preDir = r'E:\example\Weibo_data\code1\code\output'
    bc = Bert_Class(model_preDir)
    sentence = '参观了京都岚山下 世界上唯一一个守护头发的神秘地方 也和日本女性探讨了如何用碧浪守护家庭安全感的心路历程'
    result = bc.predict_on_pb(sentence)
    print(result)
