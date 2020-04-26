# coding=utf-8

import os, csv, random, collections, pickle
from bert import modeling
from bert import optimization
from bert import tokenization
from arguments import  *
import tensorflow as tf

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SelfProcessor(DataProcessor):
    """Processor for the FenLei data set (GLUE version)."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line)
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.tsv')  # cnews.train.txt
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.seed(0)
        random.shuffle(reader)  # 注意要shuffle

        examples, self.labels = [], []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")

            if len(split_line) == 2:
                text_a = tokenization.convert_to_unicode(split_line[1])
                text_b = None
                label = split_line[0]
                examples.append(InputExample(guid=guid, text_a=text_a,
                                            text_b=text_b, label=label))
                self.labels.append(label)
        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.tsv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        random.shuffle(reader)

        examples = []
        for index, line in enumerate(reader):
            guid = 'dev-%d' % index
            split_line = line.strip().split("\t")
            if len(split_line) == 2:
                text_a = tokenization.convert_to_unicode(split_line[1])
                text_b = None
                label = split_line[0]
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b=text_b, label=label))

        return examples

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.tsv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        # random.shuffle(reader)  # 测试集不打乱数据，便于比较

        examples = []
        for index, line in enumerate(reader):
            guid = 'test-%d' % index
            split_line = line.strip().split("\t")

            if len(split_line) == 2:
                text_a = tokenization.convert_to_unicode(split_line[1])
                text_b = None
                label = split_line[0]
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b=text_b, label=label))
        return examples

    def one_example(self, sentence):
        guid, label = 'pred-0', self.labels[0]
        text_a, text_b = sentence, None
        return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        self.labels = [str(x) for x in range(21)]
        return  self.labels
        # return sorted(set(self.labels), key=self.labels.index)  # 使用有序列表而不是集合。保证了标签正确



def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids, input_mask=input_mask,
        segment_ids=segment_ids, label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            #用于构造每个样本的信息键值对
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = arg_dic['train_batch_size']  # params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config, is_training=is_training,
        input_ids=input_ids, input_mask=input_mask,
        token_type_ids=segment_ids, use_one_hot_embeddings=False)

    # In the demo, we are doing a simple classification task on the entire segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output() instead.
    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train, num_warmup, ):
    """Returns `model_fn` closure for GPU Estimator."""

    def model_gpu(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for GPU 版本的 Estimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train, num_warmup, False)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op, )
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {"eval_accuracy": accuracy, "eval_loss": loss, }

            metrics = metric_fn(per_example_loss, label_ids, logits, True)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"probabilities": probabilities}, )
        return output_spec

    return model_gpu


# This function is not used by this file but is still used by the Colab and people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = 200  # params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(all_input_ids, shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "input_mask":
                tf.constant(all_input_mask, shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "segment_ids":
                tf.constant(all_segment_ids, shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def create_classification_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )

    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)



def bert_main():

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"cnews": SelfProcessor}

    tokenization.validate_case_matches_checkpoint(arg_dic['do_lower_case'], arg_dic['init_checkpoint'])

    if not arg_dic['do_train'] and not arg_dic['do_eval'] and not arg_dic['do_predict']:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])

    if arg_dic['max_seq_length'] > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (arg_dic['max_seq_length'], bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(arg_dic['output_dir'])
    tf.gfile.MakeDirs(arg_dic['pb_model_dir'])
    tf.gfile.MakeDirs(arg_dic['eval_dir'])
    task_name = arg_dic['task_name'].lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    tokenizer = tokenization.FullTokenizer(vocab_file=arg_dic['vocab_file'], do_lower_case=arg_dic['do_lower_case'])
    run_config = tf.estimator.RunConfig(model_dir=arg_dic['output_dir'],
                                        save_checkpoints_steps=arg_dic['save_checkpoints_steps'], )

    processor = processors[task_name]()
    train_examples = processor.get_train_examples(arg_dic['data_dir'])
    global label_list
    label_list = processor.get_labels()
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(os.path.join(arg_dic['eval_dir'],'label_list.pkl') , 'wb') as f:
        pickle.dump(label_list, f)
    with open(os.path.join(arg_dic['eval_dir'],'label2id.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
    num_train_steps = int(
        len(train_examples) / arg_dic['train_batch_size'] * arg_dic['num_train_epochs']) if arg_dic[
        'do_train'] else None
    num_warmup_steps = int(num_train_steps * arg_dic['warmup_proportion']) if arg_dic['do_train'] else None

    model_fn = model_fn_builder(bert_config=bert_config, num_labels=len(label_list),
                                init_checkpoint=arg_dic['init_checkpoint'], learning_rate=arg_dic['learning_rate'],
                                num_train=num_train_steps, num_warmup=num_warmup_steps)
    '''
    tf.estimator.Estimator(model_fn #模型函数, config= 设置参数对象, params= 超参数,将传递给model_fn使用,warm_start_from #热启动目录路径 )
    '''
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    if arg_dic['do_train']:
        train_file = os.path.join(arg_dic['output_dir'], "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, arg_dic['max_seq_length'], tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", arg_dic['train_batch_size'])
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file, seq_length=arg_dic['max_seq_length'],
            is_training=True, drop_remainder=True)
        try:
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        except ValueError:
            msg = '模型训练失败'
            print(msg)

    if arg_dic['do_eval']:
        eval_examples = processor.get_dev_examples(arg_dic['data_dir'])

        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(arg_dic['eval_dir'], "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, arg_dic['max_seq_length'], tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", arg_dic['eval_batch_size'])

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file, seq_length=arg_dic['max_seq_length'],
            is_training=False, drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn, )

        output_eval_file = os.path.join(arg_dic['eval_dir'], "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))




