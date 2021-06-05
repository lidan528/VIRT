#! usr/bin/env python3

# -*- coding:utf-8 -*-

import os
import codecs
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import json

import pickle
import csv

import modeling, tokenization, optimization
import collections
import random
import numpy as np
import math

import sys

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
    "max_seq_length", 128,
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

flags.DEFINE_integer(
    "max_seq_length_query", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "max_seq_length_doc", 317,    # 384-64-3 in bert
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, label_id):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a

        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b

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


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsnl(os.path.join(data_dir, "mnli-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsnl(os.path.join(data_dir, "mnli-dev.jsonl")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsnl(os.path.join(data_dir, "mnli-test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        # return ["contradiction", "entailment", "neutral"]
        return [0, 1, 2]

    def _read_jsnl(self, jsn_file):
        lines = []
        with open(jsn_file, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    line = json.loads(line)
                    lines.append(line)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #   continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line['seq1'])
            text_b = tokenization.convert_to_unicode(line['seq2'])
            # if set_type == "test":
            #  label = "contradiction"
            # else:
            # label = tokenization.convert_to_unicode(line['label']['cls'])
            label = line['label']['cls']
            # if label == tokenization.convert_to_unicode("contradictory"):
            #   label = tokenization.convert_to_unicode("contradiction")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsnl(os.path.join(data_dir, "qqp-train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsnl(os.path.join(data_dir, "qqp-dev.jsonl")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsnl(os.path.join(data_dir, "qqp-test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        # return ["contradiction", "entailment", "neutral"]
        return [0, 1]

    def _read_jsnl(self, jsn_file):
        lines = []
        with open(jsn_file, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if line:
                    line = json.loads(line)
                    lines.append(line)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #   continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line['seq1'])
            text_b = tokenization.convert_to_unicode(line['seq2'])
            # if set_type == "test":
            #  label = "contradiction"
            # else:
            # label = tokenization.convert_to_unicode(line['label']['cls'])
            label = line['label']['cls']
            # if label == tokenization.convert_to_unicode("contradictory"):
            #   label = tokenization.convert_to_unicode("contradiction")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


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
                label = tokenization.convert_to_unicode(line[2])
            #     if len(line) < 2:
            #         raise ValueError("Relation predict file columns must be 2")
            # else:
            # if len(line) < 2:
            # raise ValueError("The {0} colums must be 5".format(i))
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            # label = tokenization.convert_to_unicode(line[2])

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_single_example(ex_index, example, rele_label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(rele_label_list):
        label_map[label] = i

    # tokens_a = tokenizer.tokenize(example.text_a)
    # tokens_b = None
    # if example.text_b:
    #     tokens_b = tokenizer.tokenize(example.text_b)
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[0:(max_seq_length - 2)]

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

    # def build_bert_input(tokens_temp):
    #     tokens_p = []
    #     segment_ids = []
    #
    #     tokens_p.append("[CLS]")
    #     segment_ids.append(0)
    #
    #     for token in tokens_temp:
    #         tokens_p.append(token)
    #         segment_ids.append(0)
    #
    #     tokens_p.append("[SEP]")
    #     segment_ids.append(0)
    #
    #     input_ids = tokenizer.convert_tokens_to_ids(tokens_p)
    #     input_mask = [1] * len(input_ids)
    #
    #     while len(input_ids) < max_seq_length:
    #         input_ids.append(0)
    #         input_mask.append(0)
    #         segment_ids.append(0)
    #
    #     assert len(input_ids) == max_seq_length
    #     assert len(input_mask) == max_seq_length
    #     assert len(segment_ids) == max_seq_length
    #
    #     if ex_index < 5:
    #         tf.logging.info("*** Example ***")
    #         tf.logging.info("guid: %s" % (example.guid))
    #         tf.logging.info("tokens: %s" % " ".join(
    #             [tokenization.printable_text(x) for x in tokens_p]))
    #
    #         tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #         tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #         tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #
    #
    #     return input_ids,input_mask,segment_ids

    def build_bert_input_s_bert(tokens_temp):

        if len(tokens_temp) > max_seq_length - 2:
            tokens_temp = tokens_temp[0: (max_seq_length - 2)]

        tokens_p = []
        segment_ids = []
        tokens_p.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_temp:
            tokens_p.append(token)
            segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_p)  # [CLS], a,a,a
        input_mask = [1] * len(input_ids)  # [CLS], a,a,a

        while len(input_ids) < max_seq_length - 1:  # [CLS], a,a,a,<PAD>,
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        tokens_p.append("[SEP]")
        input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], a,a,a,<PAD>, [SEP]
        input_mask.append(1)
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

    input_ids_a, input_mask_a, segment_ids_a = build_bert_input_s_bert(tokens_a)
    input_ids_b, input_mask_b, segment_ids_b = build_bert_input_s_bert(tokens_b)

    label_id = label_map[example.label]

    feature = InputFeatures(input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b,
                            label_id)

    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, is_training=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    if is_training:
        tf.logging.info("training! write data,shuffling")
        random.shuffle(examples)

    count = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        if not feature:
            continue

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids_a"] = create_int_feature(feature.input_ids_a)
        features["input_mask_a"] = create_int_feature(feature.input_mask_a)
        features["segment_ids_a"] = create_int_feature(feature.segment_ids_a)

        features["input_ids_b"] = create_int_feature(feature.input_ids_b)
        features["input_mask_b"] = create_int_feature(feature.input_mask_b)
        features["segment_ids_b"] = create_int_feature(feature.segment_ids_b)

        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        count += 1

    tf.logging.info("proprecessd actual number of tfrecord: {0}".format(count))


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids_a": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids_a": tf.FixedLenFeature([seq_length], tf.int64),

        "input_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask_b": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids_b": tf.FixedLenFeature([seq_length], tf.int64),
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


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, use_one_hot_embeddings, is_reuse):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="bert",
        is_reuse=is_reuse
    )

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

        # token_embedding_sum = tf.reduce_sum(output_layer, 1)  # batch*hidden_size
        # output_layer = token_embedding_sum/FLAGS.max_seq_length
        mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
        output_layer = sum_masked_output_layer / actual_token_nums
    else:
        tf.logging.info("pooling_strategy error")
        assert 1 == 2

    return output_layer


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
            input_ids_a = features["input_ids_a"]
            input_mask_a = features["input_mask_a"]
            segment_ids_a = features["segment_ids_a"]
            input_ids_b = features["input_ids_b"]
            input_mask_b = features["input_mask_b"]
            segment_ids_b = features["segment_ids_b"]
            label_ids = features["label_ids"]
            query_embedding = create_model(bert_config, is_training, input_ids_a, input_mask_a, segment_ids_a,
                                           use_one_hot_embeddings, tf.AUTO_REUSE)
            doc_embedding = create_model(bert_config, is_training, input_ids_b, input_mask_b, segment_ids_b,
                                         use_one_hot_embeddings, tf.AUTO_REUSE)
        else:
            input_ids_a = features["input_ids_a"]
            input_mask_a = features["input_mask_a"]
            segment_ids_a = features["segment_ids_a"]
            query_embedding = create_model(bert_config, is_training, input_ids_a, input_mask_a, segment_ids_a,
                                           use_one_hot_embeddings, tf.AUTO_REUSE)
            doc_embedding = create_model(bert_config, is_training, input_ids_a, input_mask_a, segment_ids_a,
                                         use_one_hot_embeddings, tf.AUTO_REUSE)
            label_ids = 0
        # if mode == tf.estimator.ModeKeys.PREDICT and "id" in features:
        #    query_id = features["id"]

        sub_embedding = tf.abs(query_embedding - doc_embedding)
        max_embedding = tf.square(tf.reduce_max([query_embedding, doc_embedding], axis=0))

        regular_embedding = tf.concat([query_embedding, doc_embedding, sub_embedding, max_embedding], -1)
        logits = tf.layers.dense(regular_embedding, units=num_rele_label)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(label_ids, depth=num_rele_label, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        regular_loss = tf.reduce_mean(per_example_loss)

        # (ner_loss, _, _, pred_ids) = ner_part
        # (rele_loss, per_example_loss, _, probabilities) = rele_part

        total_loss = regular_loss
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

            def metric_fn(per_example_loss, label_ids, probabilities):
                predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                precision = tf.metrics.precision(label_ids, predictions)
                recall = tf.metrics.recall(label_ids, predictions)
                # f1 = tf.metrics.f1(label_ids, predictions, num_rele_label, [1], average="macro")

                # get positive score for auc
                auc = tf.metrics.auc(label_ids, probabilities[:, -1])

                # loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_pre": precision,
                    "eval_rec": recall,
                    #   "eval_f1": f1,
                    "eval_accuracy": accuracy,
                    "eval_auc": auc,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, probabilities])
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


def create_model_metric_mnli(bert_config, input_ids_a_ph, input_masks_a_ph, cached_emb_b,  num_labels, first_m):
    """
    只有a需要处理输入, b端直接用已有的缓存，并且由于是poly-encoder，这里的emb_b都是token粒度的
    """
    def dot_attention(q, k, v, v_mask=None, dropout=None):
        # v_mask [B, T]
        attention_scores = tf.matmul(q, k, transpose_b=True)
        # attention_scores [B, S, T]
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(bert_config.hidden_size)))
        if v_mask is not None:
            v_mask = tf.expand_dims(v_mask, axis=[1])
            adder = (1.0 - tf.cast(v_mask, tf.float32)) * -10000.0
            attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)  # [B, S, T]
        output = tf.matmul(attention_probs, v)
        return output

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids_a_ph,
        use_one_hot_embeddings=FLAGS.use_tpu)
    if FLAGS.pooling_strategy == "cls":
        tf.logging.info("use cls embedding")
        output_layer_a = model.get_pooled_output()

    elif FLAGS.pooling_strategy == "mean":
        tf.logging.info("use mean embedding")

        output_layer = model.get_sequence_output()

        mask = tf.cast(tf.expand_dims(input_masks_a_ph, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_masks_a_ph, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
        output_layer_a = sum_masked_output_layer / actual_token_nums

    query_embedding = tf.expand_dims(output_layer_a, axis=[1])
    poly_mask = tf.constant(value=np.ones((FLAGS.train_batch_size, first_m)))
    final_vecs = dot_attention(query_embedding, cached_emb_b, cached_emb_b, v_mask=poly_mask)
    final_vecs = tf.squeeze(final_vecs, axis=[1])

    logits = tf.layers.dense(final_vecs, units=num_labels)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities


def create_model_metric_squad(bert_config, input_ids_ph, input_masks_ph, cached_text_embed, num_labels, first_m):
    """
    问题需要运行，所有答案的token都为cached的Embedding
    """
    def dot_attention(q, k, v, v_mask=None, dropout=None):
        # v_mask [B, T]
        attention_scores = tf.matmul(q, k, transpose_b=True)
        # attention_scores [B, S, T]
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(bert_config.hidden_size)))
        if v_mask is not None:
            v_mask = tf.expand_dims(v_mask, axis=[1])
            adder = (1.0 - tf.cast(v_mask, tf.float32)) * -10000.0
            attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)  # [B, S, T]
        output = tf.matmul(attention_probs, v)
        return output

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids_ph,
        use_one_hot_embeddings=FLAGS.use_tpu)
    if FLAGS.pooling_strategy == "cls":
        tf.logging.info("use cls embedding")
        output_layer_a = model.get_pooled_output()

    elif FLAGS.pooling_strategy == "mean":
        tf.logging.info("use mean embedding")

        output_layer = model.get_sequence_output()

        mask = tf.cast(tf.expand_dims(input_masks_ph, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_masks_ph, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
        output_layer_a = sum_masked_output_layer / actual_token_nums

    query_embedding = tf.expand_dims(output_layer_a, axis=[1])
    poly_mask = tf.constant(value=np.ones((FLAGS.train_batch_size, first_m)))
    final_vecs = dot_attention(query_embedding, cached_text_embed[:, 0:first_m, :], cached_text_embed[:, 0:first_m, :], v_mask=poly_mask)
    final_vecs = tf.squeeze(final_vecs, axis=[1])

    pooled_output_layer_query = tf.expand_dims(final_vecs, axis=1)  # [bs, 1, emb_dim]
    pooled_output_layer_query = tf.tile(pooled_output_layer_query, [1, FLAGS.max_seq_length_doc, 1])
    sub_embedding = tf.abs(pooled_output_layer_query - cached_text_embed)
    max_embedding = tf.square(tf.reduce_max([pooled_output_layer_query, cached_text_embed], axis=0))
    regular_embedding = tf.concat([pooled_output_layer_query, cached_text_embed, sub_embedding, max_embedding], -1)
    # #[bs, seq_length_doc, emb_dim]

    output_weights = tf.get_variable(
        "output_weights", [2, bert_config.hidden_size * 4],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(regular_embedding,
                                     [FLAGS.train_batch_size * FLAGS.max_seq_length_doc, bert_config.hidden_size * 4])  # regular_embedding为四个向量拼接
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [FLAGS.train_batch_size, FLAGS.max_seq_length_doc, 2])
    logits = tf.transpose(logits, [2, 0, 1])  # [2, bs, seq_len]      # each position word_embedding mapped to a value

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
    return (start_logits, end_logits)  # [bs, seq_len]



def metric_flops(bert_config):
    run_metadata = tf.RunMetadata()
    processors = {
        "mnli": MnliProcessor,
        "qqp": QqpProcessor
    }
    metric_funcs = {
        "mnli": create_model_metric_mnli,
        "squad": create_model_metric_squad,
        "boolq": create_model_metric_mnli,
    }

    task_name = FLAGS.task_name.lower()
    processor = processors[task_name]() if task_name in processors else None
    metric_func = metric_funcs[task_name]
    label_list = processor.get_labels() if task_name in processors else [0]

    if task_name == 'squad':
        first_m = 16
        input_ids_a_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_query], dtype=tf.int32,
                                        name='input/input_ids')
        input_masks_a_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_query], dtype=tf.int32,
                                          name='input/input_masks')
        cached_embd_b_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_doc, bert_config.hidden_size], dtype=tf.float32,
                                          name='input/cached_emd_b')
        result = metric_func(bert_config, input_ids_a_ph, input_masks_a_ph, cached_embd_b_ph,
                                          len(label_list), first_m=first_m)
    elif task_name == 'boolq':
        input_ids_a_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_query], dtype=tf.int32,
                                        name='input/input_ids')
        input_masks_a_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_query], dtype=tf.int32,
                                          name='input/input_masks')
        cached_embd_b_ph = tf.placeholder(shape=[FLAGS.train_batch_size, bert_config.hidden_size], dtype=tf.float32,
                                          name='input/cached_emd_b')
        result = metric_func(bert_config, input_ids_a_ph, input_masks_a_ph, cached_embd_b_ph, len(label_list))

    else:
        first_m = 360
        input_ids_a_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_query], dtype=tf.int32, name='input/input_ids')
        input_masks_a_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length_query], dtype=tf.int32, name='input/input_masks')
        cached_embd_b_ph = tf.placeholder(shape=[FLAGS.train_batch_size, first_m, bert_config.hidden_size], dtype=tf.float32, name='input/cached_emd_b')
        result = metric_func(bert_config, input_ids_a_ph, input_masks_a_ph, cached_embd_b_ph, len(label_list), first_m=first_m)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        opt_builder = tf.profiler.ProfileOptionBuilder
        prof_options = opt_builder.float_operation()
        prof_options['hide_name_regexes'] = ['.*/Initializer/.*']
        tfprof_node = tf.profiler.profile(sess.graph, options=prof_options)
        tf.logging.info('GFLOPs: {};    '.format(tfprof_node.total_float_ops / 1000000000.0))



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    metric_flops(bert_config)


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
