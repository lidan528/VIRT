#! usr/bin/env python3

# -*- coding:utf-8 -*-

import os
import codecs
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

import pickle
import csv

import modeling, tokenization, optimization
import collections
import random

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("data_dir", None, "data path")

flags.DEFINE_string(
    "test_flie_name", None,
    "test_file_name.")

flags.DEFINE_string(
    "train_data_path", None,
    "The input data path. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "eval_data_path", None,
    "The input data path. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "predict_data_path", None,
    "The input data path. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length_a", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_b", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("log_step_count_steps", 50, "output log every x steps")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 64, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("max_eval_steps", 20, "Maximum number of eval steps.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 200,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("ner_label_file", None, "ner label files")

flags.DEFINE_string("pooling_strategy", "cls", "Pooling Strategy")

flags.DEFINE_bool("do_save", False, "Whether to save the checkpoint to pb")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a1, text_b1=None, text_a2=None, text_b2=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        # self.guid = guid
        # self.text_a = text_a
        # self.text_b = text_b
        # self.label = label
        self.guid = guid
        self.text_a1 = text_a1
        self.text_b1 = text_b1
        self.text_a2 = text_a2
        self.text_b2 = text_b2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_a1, input_mask_a1, segment_ids_a1,
                 input_ids_b1, input_mask_b1, segment_ids_b1,
                 input_ids_a2, input_mask_a2, segment_ids_a2,
                 input_ids_b2, input_mask_b2, segment_ids_b2, label_id):
        self.input_ids_a1 = input_ids_a1
        self.input_mask_a1 = input_mask_a1
        self.segment_ids_a1 = segment_ids_a1

        self.input_ids_b1 = input_ids_b1
        self.input_mask_b1 = input_mask_b1
        self.segment_ids_b1 = segment_ids_b1

        self.input_ids_a2 = input_ids_a2
        self.input_mask_a2 = input_mask_a2
        self.segment_ids_a2 = segment_ids_a2

        self.input_ids_b2 = input_ids_b2
        self.input_mask_b2 = input_mask_b2
        self.segment_ids_b2 = segment_ids_b2

        self.label_id = label_id


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
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in f:
                line = line.replace('\n', '')
                line = line.replace('\x00', '')
                lines.append(line.split('\t'))
            return lines

    @classmethod
    def _horovod_read_file_single_mode(cls, input_file, quotechar):
        with tf.gfile.GFile(input_file, "r") as reader:
            lines = []
            while True:
                line = reader.readline()
                if not line:
                    break
                attr = line.strip().split('\t')
                lines.append(attr)
            return lines


class LcqmcProcessor(DataProcessor):
    """Processor for the meituan data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if len(lines) <= 0:
            raise ValueError("Must specify file size > 0")
        for (i, line) in enumerate(lines):
            if i % 10000 == 0:
                tf.logging.info("reading example %d of %d" % (i, len(lines)))
            if set_type == "test":
                label = "0"
            else:
                # label = tokenization.convert_to_unicode(line[2])
                label = "1"
            #     if len(line) < 2:
            #         raise ValueError("Relation predict file columns must be 2")
            # else:
            # if len(line) < 2:
            # raise ValueError("The {0} colums must be 5".format(i))
            guid = "%s-%s" % (set_type, i)
            text_a1 = tokenization.convert_to_unicode(line[0])
            text_b1 = tokenization.convert_to_unicode(line[1])

            text_a2 = tokenization.convert_to_unicode(line[0])
            text_b2 = tokenization.convert_to_unicode(line[2])

            # text_a = tokenization.convert_to_unicode(line[0])
            # text_b = tokenization.convert_to_unicode(line[1])

            # label = tokenization.convert_to_unicode(line[2])

            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,label=label))
            examples.append(
                InputExample(guid=guid, text_a1=text_a1, text_b1=text_b1, text_a2=text_a2, text_b2=text_b2, label=label))
        return examples


def convert_single_example(ex_index, example, text_a, text_b, rele_label_list, max_seq_length_a, max_seq_length_b,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(rele_label_list):
        label_map[label] = i

    # tokens_a = tokenizer.tokenize(example.text_a)
    # tokens_b = None
    # if example.text_b:
    #     tokens_b = tokenizer.tokenize(example.text_b)
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    if len(tokens_a) > max_seq_length_a - 2:
        tokens_a = tokens_a[0:(max_seq_length_a - 2)]

    if len(tokens_b) > max_seq_length_b - 2:
        tokens_b = tokens_b[0:(max_seq_length_b - 2)]

    # actual_a = len(tokens_a)
    # actual_b = len(tokens_b)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    def build_bert_input(tokens_temp, max_seq_length):
        tokens_p = []
        segment_ids = []

        tokens_p.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_temp:
            tokens_p.append(token)
            segment_ids.append(0)

        tokens_p.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens_p)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens_p]))

            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        return input_ids, input_mask, segment_ids

    input_ids_a, input_mask_a, segment_ids_a = build_bert_input(tokens_a, max_seq_length_a)
    input_ids_b, input_mask_b, segment_ids_b = build_bert_input(tokens_b, max_seq_length_b)

    # label_id = label_map[example.label]

    # if ex_index < 5:
    #     tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    # feature = InputFeatures(input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, label_id)

    # return feature
    return input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b


def convert_single_example_pair(ex_index, example, label_list, max_seq_length_a, max_seq_length_b,
                                tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    label_id = label_map[example.label]

    (input_ids_a1, input_mask_a1, segment_ids_a1, input_ids_b1, input_mask_b1, segment_ids_b1) = convert_single_example(
        ex_index, example, example.text_a1, example.text_b1, label_list, max_seq_length_a, max_seq_length_b, tokenizer)
    (input_ids_a2, input_mask_a2, segment_ids_a2, input_ids_b2, input_mask_b2, segment_ids_b2) = convert_single_example(
        ex_index, example, example.text_a2, example.text_b2, label_list, max_seq_length_a, max_seq_length_b, tokenizer)

    feature = InputFeatures(
        input_ids_a1=input_ids_a1,
        input_mask_a1=input_mask_a1,
        segment_ids_a1=segment_ids_a1,
        input_ids_b1=input_ids_b1,
        input_mask_b1=input_mask_b1,
        segment_ids_b1=segment_ids_b1,
        input_ids_a2=input_ids_a2,
        input_mask_a2=input_mask_a2,
        segment_ids_a2=segment_ids_a2,
        input_ids_b2=input_ids_b2,
        input_mask_b2=input_mask_b2,
        segment_ids_b2=segment_ids_b2,
        label_id=label_id
    )
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length_a, max_seq_length_b, tokenizer, output_file, is_training=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    if is_training:
        tf.logging.info("training! write data,shuffling")
        random.shuffle(examples)

    count = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example_pair(ex_index, example, label_list,
                                              max_seq_length_a, max_seq_length_b, tokenizer)

        if not feature:
            continue

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids_a1"] = create_int_feature(feature.input_ids_a1)
        features["input_mask_a1"] = create_int_feature(feature.input_mask_a1)
        features["segment_ids_a1"] = create_int_feature(feature.segment_ids_a1)

        features["input_ids_a2"] = create_int_feature(feature.input_ids_a2)
        features["input_mask_a2"] = create_int_feature(feature.input_mask_a2)
        features["segment_ids_a2"] = create_int_feature(feature.segment_ids_a2)

        features["input_ids_b1"] = create_int_feature(feature.input_ids_b1)
        features["input_mask_b1"] = create_int_feature(feature.input_mask_b1)
        features["segment_ids_b1"] = create_int_feature(feature.segment_ids_b1)

        features["input_ids_b2"] = create_int_feature(feature.input_ids_b2)
        features["input_mask_b2"] = create_int_feature(feature.input_mask_b2)
        features["segment_ids_b2"] = create_int_feature(feature.segment_ids_b2)

        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        count += 1

    tf.logging.info("proprecessd actual number of tfrecord: {0}".format(count))


def file_based_input_fn_builder(input_file, seq_length_a, seq_length_b, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids_a1": tf.FixedLenFeature([seq_length_a], tf.int64),
        "input_mask_a1": tf.FixedLenFeature([seq_length_a], tf.int64),
        "segment_ids_a1": tf.FixedLenFeature([seq_length_a], tf.int64),

        "input_ids_a2": tf.FixedLenFeature([seq_length_a], tf.int64),
        "input_mask_a2": tf.FixedLenFeature([seq_length_a], tf.int64),
        "segment_ids_a2": tf.FixedLenFeature([seq_length_a], tf.int64),

        "input_ids_b1": tf.FixedLenFeature([seq_length_b], tf.int64),
        "input_mask_b1": tf.FixedLenFeature([seq_length_b], tf.int64),
        "segment_ids_b1": tf.FixedLenFeature([seq_length_b], tf.int64),

        "input_ids_b2": tf.FixedLenFeature([seq_length_b], tf.int64),
        "input_mask_b2": tf.FixedLenFeature([seq_length_b], tf.int64),
        "segment_ids_b2": tf.FixedLenFeature([seq_length_b], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
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
        batch_size = params["batch_size"]

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


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#     """Truncates a sequence pair in place to the maximum length."""
#
#     # This is a simple heuristic which will always truncate the longer sequence
#     # one token at a time. This makes more sense than truncating an equal percent
#     # of tokens from each, since if one sequence is very short then each token
#     # that's truncated likely contains more information than a longer sequence.
#     while True:
#         total_length = len(tokens_a) + len(tokens_b)
#         if total_length <= max_length:
#             break
#         if len(tokens_a) > len(tokens_b):
#             tokens_a.pop()
#         else:
#             tokens_b.pop()


def create_model(bert_config, max_seq_length, is_training, input_ids, input_mask,
                 segment_ids, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = None

    if FLAGS.pooling_strategy == "cls":
        tf.logging.info("use cls embedding")
        output_layer = model.get_pooled_output()

    elif FLAGS.pooling_strategy == "mean":
        tf.logging.info("use mean embedding")

        output_layer = model.get_sequence_output()

        # delete cls and sep
        # a = tf.cast(tf.reduce_sum(input_mask, axis=-1) - 1, tf.int32)
        # last = tf.one_hot(a, depth=FLAGS.max_seq_length)

        # b = tf.zeros([tf.shape(input_ids)[0]], tf.int32)
        # first = tf.one_hot(b, depth=FLAGS.max_seq_length)
        # input_mask_sub2 = tf.cast(input_mask, dtype=tf.float32)
        # input_mask_sub2 = input_mask_sub2 - first - last

        # input_mask3 = tf.cast(tf.reshape(input_mask_sub2, [-1, FLAGS.max_seq_length, 1]), tf.float32)
        # output_layer = output_layer * input_mask3

        # average pooling
        # length = tf.reduce_sum(input_mask3, axis=1)
        token_embedding_sum = tf.reduce_sum(output_layer, 1)  # batch*hidden_size
        # output_layer = token_embedding_sum/length
        output_layer = token_embedding_sum / max_seq_length
    else:
        tf.logging.info("pooling_strategy error")
        assert 1 == 2

    return output_layer


def cosine_distance(query_embedding, doc_embedding):
    sub_embedding = tf.abs(query_embedding - doc_embedding)
    max_embedding = tf.square(tf.reduce_max([query_embedding, doc_embedding], axis=0))
    query_norm = tf.sqrt(tf.reduce_sum(tf.multiply(query_embedding, query_embedding), axis=-1))
    sentence_norm = tf.sqrt(tf.reduce_sum(tf.multiply(doc_embedding, doc_embedding), axis=-1))
    inner_pd = tf.reduce_sum(tf.multiply(query_embedding, doc_embedding), axis=-1)
    inner_pd_dorm = tf.multiply(query_norm, sentence_norm)
    cosine_score = tf.div(inner_pd, inner_pd_dorm + 1e-9)
    cosine_score = tf.nn.relu(cosine_score)
    logits = tf.concat([tf.expand_dims(1 - cosine_score, -1), tf.expand_dims(cosine_score, -1)], 1)
    return logits


def model_fn_builder(bert_config, num_rele_label, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            input_ids_a1 = features["input_ids_a1"]
            input_mask_a1 = features["input_mask_a1"]
            segment_ids_a1 = features["segment_ids_a1"]
            input_ids_b1 = features["input_ids_b1"]
            input_mask_b1 = features["input_mask_b1"]
            segment_ids_b1 = features["segment_ids_b1"]
            label_ids = features["label_ids"]
            query_embedding1 = create_model(bert_config, FLAGS.max_seq_length_a, is_training, input_ids_a1,
                                            input_mask_a1, segment_ids_a1, use_one_hot_embeddings)
            doc_embedding1 = create_model(bert_config, FLAGS.max_seq_length_b, is_training, input_ids_b1, input_mask_b1,
                                          segment_ids_b1, use_one_hot_embeddings)

            input_ids_a2 = features["input_ids_a2"]
            input_mask_a2 = features["input_mask_a2"]
            segment_ids_a2 = features["segment_ids_a2"]
            input_ids_b2 = features["input_ids_b2"]
            input_mask_b2 = features["input_mask_b2"]
            segment_ids_b2 = features["segment_ids_b2"]
            # label_ids = features["label_ids"]
            query_embedding2 = create_model(bert_config, FLAGS.max_seq_length_a, is_training, input_ids_a2,
                                            input_mask_a2,
                                            segment_ids_a2, use_one_hot_embeddings)
            doc_embedding2 = create_model(bert_config, FLAGS.max_seq_length_b, is_training, input_ids_b2, input_mask_b2,
                                          segment_ids_b2, use_one_hot_embeddings)
        else:
            input_ids_a = features["input_ids_a"]
            input_mask_a = features["input_mask_a"]
            segment_ids_a = features["segment_ids_a"]
            query_embedding = create_model(bert_config, FLAGS.max_seq_length_a, is_training, input_ids_a, input_mask_a,
                                           segment_ids_a, use_one_hot_embeddings)
            doc_embedding = create_model(bert_config, FLAGS.max_seq_length_b, is_training, input_ids_a, input_mask_a,
                                         segment_ids_a, use_one_hot_embeddings)
            label_ids = 0
        # if mode == tf.estimator.ModeKeys.PREDICT and "id" in features:
        #    query_id = features["id"]

        logits1 = cosine_distance(query_embedding=query_embedding1, doc_embedding=doc_embedding1)
        scores1 = logits1[:, 1]
        logits2 = cosine_distance(query_embedding=query_embedding2, doc_embedding=doc_embedding2)
        scores2 = logits2[:, 1]
        pred = 1 / (1 + tf.exp(-(scores1 - scores2)))
        tf.logging.info("pred= %s", pred.shape)
        cross_entropy = -tf.log(pred)  # label = 1
        tf.logging.info("cse= %s", cross_entropy.shape)
        total_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy))

        # tf.logging.info("loss = %s", total_loss.shape)

        # probabilities = logits
        #
        # #regular_embedding = tf.concat([query_embedding, doc_embedding, sub_embedding, max_embedding], -1)
        # #logits = tf.layers.dense(regular_embedding, units=num_rele_label)
        #
        # #probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(label_ids, depth=num_rele_label, dtype=tf.float32)
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # regular_loss = tf.reduce_mean(per_example_loss)

        # (ner_loss, _, _, pred_ids) = ner_part
        # (rele_loss, per_example_loss, _, probabilities) = rele_part

        # total_loss = regular_loss
        # total_loss = rele_loss
        # total_loss = 0.1*ner_loss + rele_loss

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(cross_entropy, label_ids, probabilities1, probabilities2):
                # predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
                predictions = tf.expand_dims(tf.cast(probabilities1[:, -1] > probabilities2[:, -1], dtype=tf.float32), -1)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                precision = tf.metrics.precision(label_ids, predictions)
                recall = tf.metrics.recall(label_ids, predictions)
                loss = tf.reduce_mean(tf.reduce_sum(cross_entropy))
                #f1 = tf.metrics.f1(label_ids, predictions, num_rele_label, [1], average="macro")

                #get positive score for auc
                auc = tf.metrics.auc(label_ids, predictions)

                # loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_pre": precision,
                    "eval_rec": recall,
                 #   "eval_f1": f1,
                    "eval_accuracy": accuracy,
                    "eval_auc": auc,
                   # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [cross_entropy, label_ids, logits1, logits2])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=query_embedding,
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def serving_input_receiver_fn(max_seq_length):
    input_ids_a = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length], name="input_ids_a")
    # input_ids_b = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length], name="input_ids_b")
    input_mask_a = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length], name="input_mask_a")
    # input_mask_b = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length], name="input_mask_b")
    segment_ids_a = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_length], name="segment_ids_a")

    feature_spec = {'input_ids_a': input_ids_a, 'input_mask_a': input_mask_a, 'segment_ids_a': segment_ids_a, }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "lcqmc": LcqmcProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_save:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length_a > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length_a, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length_b > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length_b, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_rele_label=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # # If TPU is not available, this will fall back to normal Estimator on CPU
    # # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # def file_based_convert_examples_to_features(
    #         examples, ner_label_map, label_list, max_seq_length, tokenizer, output_file, is_training=False):

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if tf.gfile.Exists(train_file):
            print("train file exists")
        else:
            file_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length_a, FLAGS.max_seq_length_b, tokenizer, train_file,
                is_training=True)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length_a=FLAGS.max_seq_length_a,
            seq_length_b=FLAGS.max_seq_length_b,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length_a, FLAGS.max_seq_length_b, tokenizer, eval_file)
        else:
            print("eval file exists")

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length_a=FLAGS.max_seq_length_a,
            seq_length_b=FLAGS.max_seq_length_b,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                if key.startswith("eval"):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            # pre = result["eval_pre"]
            # rec = result["eval_rec"]
            # if pre == 0 and rec == 0:
            #     fscore = 0
            # else:
            #     fscore = 2 * pre * rec / (pre + rec)
            # writer.write("%s = %s\n" % ("eval_fscore", str(fscore)))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir, FLAGS.test_flie_name)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        if not tf.gfile.Exists(predict_file):
            file_based_convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length_a,
                                                    FLAGS.max_seq_length_b, tokenizer, predict_file)
        else:
            print("predict file Exists")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length_a=FLAGS.max_seq_length_a,
            seq_length_b=FLAGS.max_seq_length_b,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        eval_steps = None
        # test_result = estimator.evaluate(input_fn=predict_input_fn, steps=eval_steps)
        # output_test_file = os.path.join(FLAGS.output_dir, "test_results.txt")
        # with tf.gfile.GFile(output_test_file, "w") as writer:
        # tf.logging.info("***** test results *****")
        # for key in sorted(test_result.keys()):
        # tf.logging.info("  %s = %s", key, str(test_result[key]))
        # writer.write("%s = %s\n" % (key, str(test_result[key])))

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "predict_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                # output_line = str(prediction[1]) + "\n"
                output_line = ",".join(str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)

    if FLAGS.do_save:
        estimator._export_to_tpu = False
        estimator.export_savedmodel(FLAGS.output_dir,
                                    serving_input_receiver_fn=serving_input_receiver_fn(FLAGS.max_seq_length_a))
        tf.logging.info("******* Done for exporting pb file***********")


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
