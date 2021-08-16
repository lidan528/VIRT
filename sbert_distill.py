#! usr/bin/env python3

# -*- coding:utf-8 -*-

import os
import codecs
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

import pickle
import csv
import json
import math

import modeling, tokenization, optimization
import collections
import random

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
    "init_checkpoint_teacher", None,
    "Initial checkpoint for teacher.")

flags.DEFINE_string(
    "init_checkpoint_student", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length_bert", 130,
    "The maximum total input sequence length after WordPiece tokenization. "
    "In the origin bert model"
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_sbert", 259,
    "The maximum total input sequence length after WordPiece tokenization. "
    "In the s-bert model."
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_float(
    "kd_weight_logit", None,
    "The weight loss of kd logits."
)

flags.DEFINE_bool(
    "use_kd_logit_mse", None,
    "Whether to use logit kl distillations"
)

flags.DEFINE_bool(
    "use_layer_distill",  False,
    "whether to use distill for hidden layer."
)

flags.DEFINE_string(
    "layer_distill_mode", None,
    "direct_mean or map_mean."
)

flags.DEFINE_float(
    "kd_weight_layer", 0,
    "The weight of hidden layer distillation."
)

flags.DEFINE_bool(
    "use_kd_logit_kl", None,
    "Whether to use logit mse distillations"
)

flags.DEFINE_bool(
    "use_kd_att", None,
    "Whether to use attention distillations"
)

flags.DEFINE_float(
    "kd_weight_att", None,
    "The weight of att distillation"
)

flags.DEFINE_bool(
    "use_contrast_self", None,
    "Whether to use attention distillations self"
)

flags.DEFINE_bool(
    "use_contrast_teacher_separately", None,
    "Whether to use attention distillations self"
)


flags.DEFINE_float(
    "weight_contrast", None,
    "The weight of att distillation"
)

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

flags.DEFINE_integer("save_checkpoints_steps", 500,
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

# flags.DEFINE_string("pooling_strategy", "cls", "Pooling Strategy")

flags.DEFINE_bool("do_save", False, "Whether to save the checkpoint to pb")

flags.DEFINE_string(
    "pooling_strategy", None,
    "cls or mean"
)

flags.DEFINE_bool("use_all_layer_emb", False, "Whether to use all layer embedding.")

flags.DEFINE_bool("use_resnet_predict", False, "Whether to use resnet in predict.")

flags.DEFINE_bool("use_weighted_att", False, "Whether to use weighted att distill.")


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

    def __init__(self,
                 input_ids_sbert_a, input_mask_sbert_a, segment_ids_sbert_a,
                 input_ids_sbert_b, input_mask_sbert_b, segment_ids_sbert_b,
                 input_ids_bert_ab, input_mask_bert_ab, segment_ids_bert_ab,
                 label_id):
        self.input_ids_sbert_a = input_ids_sbert_a
        self.input_mask_sbert_a = input_mask_sbert_a
        self.segment_ids_sbert_a = segment_ids_sbert_a

        self.input_ids_sbert_b = input_ids_sbert_b
        self.input_mask_sbert_b = input_mask_sbert_b
        self.segment_ids_sbert_b = segment_ids_sbert_b

        self.input_ids_bert_ab = input_ids_bert_ab
        self.input_mask_bert_ab = input_mask_bert_ab
        self.segment_ids_bert_ab = segment_ids_bert_ab

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
      #if set_type == "test":
      #  label = "contradiction"
      #else:
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
      #if set_type == "test":
      #  label = "contradiction"
      #else:
      # label = tokenization.convert_to_unicode(line['label']['cls'])
      label = line['label']['cls']
      # if label == tokenization.convert_to_unicode("contradictory"):
      #   label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


# def convert_single_example(ex_index, example, rele_label_list, max_seq_length,
#                            tokenizer):
#     """Converts a single `InputExample` into a single `InputFeatures`."""
#     label_map = {}
#     for (i, label) in enumerate(rele_label_list):
#         label_map[label] = i
#
#     # tokens_a = tokenizer.tokenize(example.text_a)
#     # tokens_b = None
#     # if example.text_b:
#     #     tokens_b = tokenizer.tokenize(example.text_b)
#     tokens_a = tokenizer.tokenize(example.text_a)
#     tokens_b = tokenizer.tokenize(example.text_b)
#
#     if len(tokens_a) > max_seq_length - 2:
#         tokens_a = tokens_a[0:(max_seq_length - 2)]
#
#     if len(tokens_b) > max_seq_length - 2:
#         tokens_b = tokens_b[0:(max_seq_length - 2)]
#
#     # actual_a = len(tokens_a)
#     # actual_b = len(tokens_b)
#
#     # The convention in BERT is:
#     # (a) For sequence pairs:
#     #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#     #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
#     # (b) For single sequences:
#     #  tokens:   [CLS] the dog is hairy . [SEP]
#     #  type_ids: 0     0   0   0  0     0 0
#     #
#     # Where "type_ids" are used to indicate whether this is the first
#     # sequence or the second sequence. The embedding vectors for `type=0` and
#     # `type=1` were learned during pre-training and are added to the wordpiece
#     # embedding vector (and position vector). This is not *strictly* necessary
#     # since the [SEP] token unambiguously separates the sequences, but it makes
#     # it easier for the model to learn the concept of sequences.
#     #
#     # For classification tasks, the first vector (corresponding to [CLS]) is
#     # used as as the "sentence vector". Note that this only makes sense because
#     # the entire model is fine-tuned.
#
#     def build_bert_input(tokens_temp):
#         tokens_p = []
#         segment_ids = []
#
#         tokens_p.append("[CLS]")
#         segment_ids.append(0)
#
#         for token in tokens_temp:
#             tokens_p.append(token)
#             segment_ids.append(0)
#
#         tokens_p.append("[SEP]")
#         segment_ids.append(0)
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens_p)
#         input_mask = [1] * len(input_ids)
#
#         while len(input_ids) < max_seq_length:
#             input_ids.append(0)
#             input_mask.append(0)
#             segment_ids.append(0)
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         if ex_index < 5:
#             tf.logging.info("*** Example ***")
#             tf.logging.info("guid: %s" % (example.guid))
#             tf.logging.info("tokens: %s" % " ".join(
#                 [tokenization.printable_text(x) for x in tokens_p]))
#
#             tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#
#         return input_ids, input_mask, segment_ids
#
#     input_ids_a, input_mask_a, segment_ids_a = build_bert_input(tokens_a)
#     input_ids_b, input_mask_b, segment_ids_b = build_bert_input(tokens_b)
#
#     label_id = label_map[example.label]
#
#     if ex_index < 5:
#         tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
#
#     feature = InputFeatures(input_ids_a, input_mask_a, segment_ids_a, input_ids_b, input_mask_b, segment_ids_b,
#                             label_id)
#
#     return feature


def conver_single_example_distill(ex_index, example, rele_label_list, max_seq_length_bert, max_seq_length_s_bert,
                           tokenizer):
    """
    由于teacher模型与student模型的输入不一致，所以需要在convert_example中同时赋予两种输入
    """
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
    #
    # if len(tokens_a) > max_seq_length_bert - 2:
    #     tokens_a = tokens_a[0:(max_seq_length_bert - 2)]
    #
    # if len(tokens_b) > max_seq_length_bert - 2:
    #     tokens_b = tokens_b[0:(max_seq_length_bert - 2)]

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
    def build_bert_input_bert(tokens_tmp_a, tokens_tmp_b):
        """
        为了能够将双塔和交互的att score map对应, 在交互中将sen1与sen2对半分开;
        在双塔模型中, 将sen1与sen2分别设置为对半分开的max_len
        """
        # 交互: [CLS] a,a,a,a,a,a, [SEP], b, b, b, b ,b [SEP]
        max_token_len = max_seq_length_bert - 3
        max_a_len = max_token_len // 2
        max_b_len = max_token_len - max_a_len
        if len(tokens_tmp_a) > max_a_len:
            tokens_tmp_a = tokens_tmp_a[0:max_a_len]
        if len(tokens_tmp_b) > max_b_len:
            tokens_tmp_b = tokens_tmp_b[0:max_b_len]
        tokens_bert_tmp = []
        segment_ids_bert = []
        tokens_bert_tmp.append("[CLS]")
        segment_ids_bert.append(0)
        for t_a in tokens_tmp_a:
            tokens_bert_tmp.append(t_a)
            segment_ids_bert.append(0)
        input_ids_bert = tokenizer.convert_tokens_to_ids(tokens_bert_tmp)
        input_masks_bert = [1]*len(input_ids_bert)          # [CLS], a,a,
        while len(input_ids_bert) < 1 + max_a_len:
            input_ids_bert.append(0)
            input_masks_bert.append(0)
            segment_ids_bert.append(0)
        tokens_bert_tmp.append("[SEP]")
        input_ids_bert += tokenizer.convert_tokens_to_ids(["[SEP]"])     #[CLS], a,a,a,<PAD>,[SEP],
        input_masks_bert.append(1)
        segment_ids_bert.append(0)
        # ---------------------------------
        tokens_bert_tmp =[]
        for t_b in tokens_tmp_b:
            tokens_bert_tmp.append(t_b)
            input_masks_bert.append(1)
            segment_ids_bert.append(1)
        input_ids_bert += tokenizer.convert_tokens_to_ids(tokens_bert_tmp)  # [CLS], a,a,a,<PAD>,[SEP], b,b,b
        while len(input_ids_bert) <  1 + max_a_len + 1 + max_b_len:
            input_ids_bert.append(0)
            input_masks_bert.append(0)
            segment_ids_bert.append(1)
        tokens_bert_tmp.append("[SEP]")
        input_ids_bert += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], a,a,a,<PAD>,[SEP], b,b,b,<PAD>, [SEP]
        input_masks_bert.append(1)
        segment_ids_bert.append(1)

        assert len(input_ids_bert) == max_seq_length_bert
        assert len(input_masks_bert) == max_seq_length_bert
        assert len(segment_ids_bert) == max_seq_length_bert

        return input_ids_bert, input_masks_bert, segment_ids_bert

    # def build_bert_input_s_bert(tokens_temp):
    #
    #     if len(tokens_temp) > max_seq_length_s_bert - 2:
    #         tokens_temp = tokens_temp[0: (max_seq_length_s_bert - 2)]
    #
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
    #     while len(input_ids) < max_seq_length_s_bert:
    #         input_ids.append(0)
    #         input_mask.append(0)
    #         segment_ids.append(0)
    #
    #     assert len(input_ids) == max_seq_length_s_bert
    #     assert len(input_mask) == max_seq_length_s_bert
    #     assert len(segment_ids) == max_seq_length_s_bert
    #
    #     return input_ids, input_mask, segment_ids

    def build_bert_input_s_bert(tokens_temp):

        if len(tokens_temp) > max_seq_length_s_bert - 2:
            tokens_temp = tokens_temp[0: (max_seq_length_s_bert - 2)]

        tokens_p = []
        segment_ids = []

        tokens_p.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_temp:
            tokens_p.append(token)
            segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_p)  # [CLS], a,a,a
        input_mask = [1] * len(input_ids)  # [CLS], a,a,a

        while len(input_ids) < max_seq_length_s_bert-1: # [CLS], a,a,a,<PAD>,
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        tokens_p.append("[SEP]")
        input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])     # [CLS], a,a,a,<PAD>, [SEP]
        input_mask.append(1)
        segment_ids.append(0)

        assert len(input_ids) == max_seq_length_s_bert
        assert len(input_mask) == max_seq_length_s_bert
        assert len(segment_ids) == max_seq_length_s_bert
        return input_ids, input_mask, segment_ids

    input_ids_bert_ab, input_mask_bert_ab, segment_ids_bert_ab = build_bert_input_bert(tokens_a, tokens_b)
    input_ids_sbert_a, input_mask_sbert_a, segment_ids_sbert_a = build_bert_input_s_bert(tokens_a)
    input_ids_sbert_b, input_mask_sbert_b, segment_ids_sbert_b = build_bert_input_s_bert(tokens_b)

    label_id = label_map[example.label]

    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in ["[CLS]"]+tokens_a+["[SEP]"]+tokens_b+["[SEP]"]]))

        tf.logging.info("input_ids_bert_ab: %s" % " ".join([str(x) for x in input_ids_bert_ab]))
        tf.logging.info("input_mask_bert_ab: %s" % " ".join([str(x) for x in input_mask_bert_ab]))
        tf.logging.info("segment_ids_bert_ab: %s" % " ".join([str(x) for x in segment_ids_bert_ab]))
        tf.logging.info("-------------------------------------------------------------------------------")
        tf.logging.info("input_ids_sbert_a: %s" % " ".join([str(x) for x in input_ids_sbert_a]))
        tf.logging.info("input_mask_sbert_a: %s" % " ".join([str(x) for x in input_mask_sbert_a]))
        tf.logging.info("segment_ids_sbert_a: %s" % " ".join([str(x) for x in segment_ids_sbert_a]))
        tf.logging.info("-------------------------------------------------------------------------------")
        tf.logging.info("input_ids_sbert_b: %s" % " ".join([str(x) for x in input_ids_sbert_b]))
        tf.logging.info("input_mask_sbert_b: %s" % " ".join([str(x) for x in input_mask_sbert_b]))
        tf.logging.info("segment_ids_sbert_b: %s" % " ".join([str(x) for x in segment_ids_sbert_b]))

    feature = InputFeatures(input_ids_sbert_a, input_mask_sbert_a, segment_ids_sbert_a,
                            input_ids_sbert_b, input_mask_sbert_b, segment_ids_sbert_b,
                            input_ids_bert_ab, input_mask_bert_ab, segment_ids_bert_ab,
                            label_id)

    return feature



def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length_bert, max_seq_length_s_bert, tokenizer, output_file, is_training=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    if is_training:
        tf.logging.info("training! write data,shuffling")
        random.shuffle(examples)

    count = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = conver_single_example_distill(ex_index, example, label_list,
                                         max_seq_length_bert, max_seq_length_s_bert, tokenizer)

        if not feature:
            continue

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids_sbert_a"] = create_int_feature(feature.input_ids_sbert_a)
        features["input_mask_sbert_a"] = create_int_feature(feature.input_mask_sbert_a)
        features["segment_ids_sbert_a"] = create_int_feature(feature.segment_ids_sbert_a)

        features["input_ids_sbert_b"] = create_int_feature(feature.input_ids_sbert_b)
        features["input_mask_sbert_b"] = create_int_feature(feature.input_mask_sbert_b)
        features["segment_ids_sbert_b"] = create_int_feature(feature.segment_ids_sbert_b)

        features["input_ids_bert_ab"] = create_int_feature(feature.input_ids_bert_ab)
        features["input_mask_bert_ab"] = create_int_feature(feature.input_mask_bert_ab)
        features["segment_ids_bert_ab"] = create_int_feature(feature.segment_ids_bert_ab)

        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        count += 1

    tf.logging.info("proprecessd actual number of tfrecord: {0}".format(count))


def file_based_input_fn_builder(input_file, seq_length_bert, seq_length_sbert,
                                is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids_sbert_a": tf.FixedLenFeature([seq_length_sbert], tf.int64),
        "input_mask_sbert_a": tf.FixedLenFeature([seq_length_sbert], tf.int64),
        "segment_ids_sbert_a": tf.FixedLenFeature([seq_length_sbert], tf.int64),

        "input_ids_sbert_b": tf.FixedLenFeature([seq_length_sbert], tf.int64),
        "input_mask_sbert_b": tf.FixedLenFeature([seq_length_sbert], tf.int64),
        "segment_ids_sbert_b": tf.FixedLenFeature([seq_length_sbert], tf.int64),

        "input_ids_bert_ab": tf.FixedLenFeature([seq_length_bert], tf.int64),
        "input_mask_bert_ab": tf.FixedLenFeature([seq_length_bert], tf.int64),
        "segment_ids_bert_ab": tf.FixedLenFeature([seq_length_bert], tf.int64),

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
def create_model_bert(bert_config, is_training, input_ids, input_mask,
                      segment_ids, use_one_hot_embeddings, scope, is_reuse):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope=scope,
        is_reuse=is_reuse)

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


        # output_layer: [bs_size, max_len, emb_dim];        input_mask: [bs_size, max_len]
        mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)     # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer       # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)    # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)# [bs_size, 1]
        output_layer = sum_masked_output_layer / actual_token_nums

        # token_embedding_sum = tf.reduce_sum(output_layer, 1)  # batch*hidden_size
        # output_layer = token_embedding_sum/length
        # output_layer = token_embedding_sum / FLAGS.max_seq_length
    else:
        tf.logging.info("pooling_strategy error")
        assert 1 == 2

    return (output_layer, model)


def create_model_sbert(bert_config, is_training, input_ids, input_mask,
                       segment_ids, use_one_hot_embeddings, scope, is_reuse):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope=scope,
        is_reuse=is_reuse)

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


        # output_layer: [bs_size, max_len, emb_dim];        input_mask: [bs_size, max_len]
        mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)     # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer       # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)    # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)# [bs_size, 1]
        output_layer = sum_masked_output_layer / actual_token_nums

        # token_embedding_sum = tf.reduce_sum(output_layer, 1)  # batch*hidden_size
        # output_layer = token_embedding_sum/length
        # output_layer = token_embedding_sum / FLAGS.max_seq_length
    else:
        tf.logging.info("pooling_strategy error")
        assert 1 == 2

    return output_layer, model


def model_fn_builder(bert_config,
                     num_rele_label,
                     init_checkpoint_teacher, init_checkpoint_student,
                     learning_rate,
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
            input_ids_sbert_a = features["input_ids_sbert_a"]
            input_mask_sbert_a = features["input_mask_sbert_a"]
            segment_ids_sbert_a = features["segment_ids_sbert_a"]

            input_ids_sbert_b = features["input_ids_sbert_b"]
            input_mask_sbert_b = features["input_mask_sbert_b"]
            segment_ids_sbert_b = features["segment_ids_sbert_b"]

            input_ids_bert_ab = features["input_ids_bert_ab"]
            input_mask_bert_ab = features["input_mask_bert_ab"]
            segment_ids_bert_ab = features["segment_ids_bert_ab"]
            label_ids = features["label_ids"]
            query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                    input_ids = input_ids_sbert_a,
                                                    input_mask = input_mask_sbert_a,
                                                    segment_ids= segment_ids_sbert_a,
                                                    use_one_hot_embeddings = use_one_hot_embeddings,
                                                    scope = "bert_student",
                                                    is_reuse = tf.AUTO_REUSE)
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                    input_ids = input_ids_sbert_b,
                                                    input_mask = input_mask_sbert_b,
                                                    segment_ids= segment_ids_sbert_b,
                                                    use_one_hot_embeddings = use_one_hot_embeddings,
                                                    scope = "bert_student",
                                                    is_reuse = tf.AUTO_REUSE)

        if FLAGS.use_all_layer_emb:
            tf.logging.info("***************use all layer embedding to predict...*****************")
            regular_embedding = use_all_layer_embedding(model_stu_query, model_stu_doc, input_mask_sbert_a, input_mask_sbert_b)
        else:
            sub_embedding = tf.abs(query_embedding - doc_embedding)
            max_embedding = tf.square(tf.reduce_max([query_embedding, doc_embedding], axis=0))
            regular_embedding = tf.concat([query_embedding, doc_embedding, sub_embedding, max_embedding], -1)

        if FLAGS.use_resnet_predict:
            tf.logging.info("*************use resnet in prediction..************************")
            logits_student, probabilities_student, log_probs_student = \
            get_prediction_student_use_resnet(regular_embedding, num_rele_label, is_training)
        else:
            logits_student, probabilities_student, log_probs_student = \
                get_prediction_student(student_output_layer=regular_embedding,
                                       num_labels=num_rele_label,
                                       is_training=is_training)
        vars_student = tf.trainable_variables()  # bert_structure: 'bert_student/...',  cls_structure: 'cls_student/..'

        teacher_output_layer, model_teacher = create_model_bert(bert_config=bert_config, is_training=False,
                                                    input_ids = input_ids_bert_ab,
                                                    input_mask = input_mask_bert_ab,
                                                    segment_ids= segment_ids_bert_ab,
                                                    use_one_hot_embeddings = use_one_hot_embeddings,
                                                    scope = "bert_teacher",
                                                    is_reuse = tf.AUTO_REUSE)
        loss_teacher, per_example_loss_teacher, logits_teacher, probabilities_teacher = \
            get_prediction_teacher(teacher_output_layer=teacher_output_layer,
                                   num_labels=num_rele_label,
                                   labels=label_ids,
                                   is_training=False)

        vars_teacher = tf.trainable_variables()             # stu + teacher
        for var_ in vars_student:
            vars_teacher.remove(var_)

        one_hot_labels = tf.one_hot(label_ids, depth=num_rele_label, dtype=tf.float32)
        per_example_loss_stu = -tf.reduce_sum(one_hot_labels * log_probs_student, axis=-1)
        regular_loss_stu = tf.reduce_mean(per_example_loss_stu)
        tf.summary.scalar("regular_loss", regular_loss_stu)
        total_loss = regular_loss_stu
        if FLAGS.use_kd_logit_mse:
            tf.logging.info('use mse of logits as distill object...')
            distill_loss_logit_mse = tf.losses.mean_squared_error(logits_teacher, logits_student)
            scaled_logit_loss = FLAGS.kd_weight_logit * distill_loss_logit_mse
            total_loss = regular_loss_stu + scaled_logit_loss
            tf.summary.scalar("logit_loss_mse", distill_loss_logit_mse)
            tf.summary.scalar("logit_loss_kl_scaled", scaled_logit_loss)
        elif FLAGS.use_kd_logit_kl:
            tf.logging.info('use KL- of logits as distill object...')
            t_value_distribution = tf.distributions.Categorical(probs=probabilities_teacher + 1e-5)
            s_value_distribution = tf.distributions.Categorical(probs=probabilities_student + 1e-5)
            distill_loss_logit_kl = tf.reduce_mean(
                tf.distributions.kl_divergence(t_value_distribution, s_value_distribution))
            scaled_logit_loss = FLAGS.kd_weight_logit * distill_loss_logit_kl
            total_loss = regular_loss_stu + scaled_logit_loss
            tf.summary.scalar("logit_loss_kl", distill_loss_logit_kl)
            tf.summary.scalar("logit_loss_kl_scaled", scaled_logit_loss)

        ## attention loss
        if FLAGS.use_kd_att:
            tf.logging.info('use att as distill object...')
            if FLAGS.use_weighted_att:
                tf.logging.info("*******************use weighted att distillation...******************")
                distill_loss_att = get_attention_loss_with_weight_head(model_student_query=model_stu_query,
                                                      model_student_doc=model_stu_doc,
                                                      model_teacher=model_teacher,
                                                      input_mask_sbert_query=input_mask_sbert_a,
                                                      input_mask_sbert_doc=input_mask_sbert_b)
            else:
                distill_loss_att = get_attention_loss(model_student_query=model_stu_query,
                                                      model_student_doc=model_stu_doc,
                                                      model_teacher=model_teacher,
                                                      input_mask_sbert_query=input_mask_sbert_a,
                                                      input_mask_sbert_doc=input_mask_sbert_b)
            scaled_att_loss = FLAGS.kd_weight_att * distill_loss_att
            total_loss = total_loss + scaled_att_loss
            tf.summary.scalar("att_loss", distill_loss_att)
            tf.summary.scalar("att_loss_scaled", scaled_att_loss)

        ## hidden h distill
        if FLAGS.use_layer_distill:
            tf.logging.info('*****use hidden layer as distill object...')
            distill_hidden_loss = get_pooled_loss(teacher_model=model_teacher,
                                          student_model_query=model_stu_query,
                                          student_model_doc=model_stu_doc,
                                          input_mask_teacher=input_mask_bert_ab,
                                          input_mask_query=input_mask_sbert_a,
                                          input_mask_doc=input_ids_sbert_b,
                                          mode=FLAGS.layer_distill_mode)
            scaled_hidden_loss = distill_hidden_loss * FLAGS.kd_weight_layer
            total_loss = total_loss + scaled_hidden_loss
            tf.summary.scalar("hidden_loss", distill_hidden_loss)
            tf.summary.scalar("hidden_loss_scaled", scaled_hidden_loss)

        # contrast loss self....
        if FLAGS.use_contrast_self:
            tf.logging.info('*****use contrast loss self...')
            distill_contrast_loss = contrastive_loss_self(teacher_model=model_teacher,
                                          query_model=model_stu_query,
                                          doc_model=model_stu_doc,
                                          input_mask_teacher=input_mask_bert_ab,
                                          input_mask_query=input_mask_sbert_a,
                                          input_mask_doc=input_ids_sbert_b,
                                          truth_labels=label_ids)
            scaled_contrast_loss = distill_contrast_loss * FLAGS.weight_contrast
            total_loss = total_loss + scaled_contrast_loss
            tf.summary.scalar("contrast_loss_self", distill_contrast_loss)
            tf.summary.scalar("contrast_loss_scaled", scaled_contrast_loss)

        # contrast loss teacher....
        if FLAGS.use_contrast_teacher_separately:
            tf.logging.info('*****use contrast loss teacher...')
            distill_contrast_loss = contrastive_loss_teacher_separately(teacher_model=model_teacher,
                                          query_model=model_stu_query,
                                          doc_model=model_stu_doc,
                                          input_mask_teacher=input_mask_bert_ab,
                                          input_mask_query=input_mask_sbert_a,
                                          input_mask_doc=input_ids_sbert_b,
                                          truth_labels=label_ids)
            scaled_contrast_loss = distill_contrast_loss * FLAGS.weight_contrast
            total_loss = total_loss + scaled_contrast_loss
            tf.summary.scalar("contrast_loss_teacher", distill_contrast_loss)
            tf.summary.scalar("contrast_loss_scaled", scaled_contrast_loss)



        # vars_teacher: bert_structure: 'bert_teacher/...',  cls_structure: 'cls_teacher/..'
        # params_ckpt_teacher: bert_structure: 'bert/...', cls_structure: '...'
        assignment_map_teacher, initialized_variable_names_teacher = \
            modeling.get_assignment_map_from_checkpoint_teacher(
                vars_teacher, init_checkpoint_teacher
            )
        tf.train.init_from_checkpoint(init_checkpoint_teacher, assignment_map_teacher)

        assignment_map_student, initialized_variable_names_student = \
            modeling.get_assignment_map_from_checkpoint_student(
                vars_student, init_checkpoint_student
            )
        tf.train.init_from_checkpoint(init_checkpoint_student, assignment_map_student)

        tf.logging.info('****-------------------------init teacher----------------------*****')
        for v_t in assignment_map_teacher:
            tf.logging.info('**initialize ${}$ in graph with checkpoint params ${}$**'.format(assignment_map_teacher[v_t],v_t))
        tf.logging.info('--------------------------------------------------------------------')

        tf.logging.info('****-------------------------init student----------------------*****')
        for v_s in assignment_map_student:
            tf.logging.info('**initialize ${}$ in graph with checkpoint params ${}$**'.format(assignment_map_student[v_s], v_s))
        tf.logging.info('--------------------------------------------------------------------')

        #
        # tvars = tf.trainable_variables()
        #
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, vars_student)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=None)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, probabilities_teacher):
                predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
                predictions_teacher = tf.argmax(probabilities_teacher, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                accuracy_teacher = tf.metrics.accuracy(label_ids, predictions_teacher)
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
                    "eval_accuracy_teacher": accuracy_teacher,
                    "eval_auc": auc,
                }

            eval_metrics = (metric_fn, [total_loss, label_ids, probabilities_student, probabilities_teacher])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=None)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=query_embedding,
                scaffold_fn=None)
        return output_spec

    return model_fn


def get_prediction_teacher(teacher_output_layer, num_labels, labels, is_training):
    """
    获取教师模型的输出，同时在定义命名空间时，需要想好教师模型加载训练好的模型的定义
    由于教师模型<classifier_bert_bipartition>在训练时,
    -------------------------------------------------------------------------------
                 | ckpt保存的参数名       |       计算图中的参数名
    --------------------------------------------------------------------------------
    BERT层参数    |   bert/....          |      bert_teacher/....
    --------------------------------------------------------------------------------
    上层分类器参数 | output_weights,_bias |      cls_teacher/output_weights, _bias
    ----------------------------------------------------------------------------------
    """
    hidden_size = teacher_output_layer.shape[-1].value
    with tf.variable_scope("cls_teacher"):

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels],
            initializer=tf.zeros_initializer())

        if is_training:
            teacher_output_layer = tf.nn.dropout(teacher_output_layer, keep_prob=0.9)

        logits = tf.matmul(teacher_output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def get_prediction_student(student_output_layer, num_labels, is_training):
    """
    获取学生模型的输出, 同时在定义命名空间时，需要想好学生模型加载BERT原始参数的对应
    -------------------------------------------------------------------------------------------
                 | bert_base的ckpt保存的参数名       |       计算图中的参数名
    -------------------------------------------------------------------------------------------
    BERT层参数    |   bert/....                     |      bert_student/....
    --------------------------------------------------------------------------------------------
    上层分类器参数 |      暂无，因此无需加载           |      cls_student/output_weights, _bias
    --------------------------------------------------------------------------------------------
    """
    hidden_size = student_output_layer.shape[-1].value
    with tf.variable_scope("cls_student"):
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        if is_training:
            student_output_layer = tf.nn.dropout(student_output_layer, keep_prob=0.9)

        logits = tf.matmul(student_output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

    return logits, probabilities, log_probs


def get_prediction_student_use_resnet(student_embedding, num_labels, is_training):
    hidden_size = student_embedding.shape[-1].value
    with tf.variable_scope("cls_student"):
        output_weights1 = tf.get_variable(
            "output_weights1", [hidden_size, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias1 = tf.get_variable(
            "output_bias1", [hidden_size], initializer=tf.zeros_initializer())

        if is_training:
            student_embedding = tf.nn.dropout(student_embedding, keep_prob=0.9)

        layer1 = tf.matmul(student_embedding, output_weights1, transpose_b=True)
        layer1 = tf.nn.bias_add(layer1, output_bias1)
        if is_training:
            layer1 = tf.nn.dropout(layer1, keep_prob=0.9)
        layer1 = layer1 + student_embedding

        output_weights2 = tf.get_variable(
            "output_weights2", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias2 = tf.get_variable(
            "output_bias2", [num_labels], initializer=tf.zeros_initializer())
        logits = tf.matmul(layer1, output_weights2, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias2)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

    return logits, probabilities, log_probs



def create_att_mask(input_mask):
    """
    相当于将input_mask列表复制seq_len次
    得到的矩阵中, 如果原input_mask的第i个元素为0，那么矩阵的第i列就全为0
    从而防止input_mask为0的元素被att，但它可以att to别的元素
    """
    to_shape = modeling.get_shape_list(input_mask, expected_rank=2)     #[batch-size, seq_len]
    batch_size = to_shape[0]
    seq_length = to_shape[1]
    mask = tf.cast(
        tf.reshape(input_mask, [batch_size, 1, seq_length]), tf.float32)
    broadcast_ones = tf.ones(
        shape=[batch_size, seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * mask
    return mask



def get_attention_loss(model_student_query, model_student_doc, model_teacher,
                       input_mask_sbert_query, input_mask_sbert_doc):
    """
    获取交互的attention loss
    """
    tea_all_att_scores_before_mask, tea_all_att_probs_ori, \
    tea_all_q_w_4d, tea_all_k_w_4d = model_teacher.all_attention_scores_before_mask, model_teacher.all_attention_probs_ori, \
                                     model_teacher.all_q_w_4d, model_teacher.all_k_w_4d

    stu_qu_all_att_scores_before_mask, stu_qu_all_att_probs_ori, \
    stu_qu_all_q_w_4d, stu_qu_all_k_w_4d = model_student_query.all_attention_scores_before_mask, model_student_query.all_attention_probs_ori, \
                                     model_student_query.all_q_w_4d, model_student_query.all_k_w_4d

    stu_do_all_att_scores_before_mask, stu_do_all_att_probs_ori, \
    stu_do_all_q_w_4d, stu_do_all_k_w_4d = model_student_doc.all_attention_scores_before_mask, model_student_doc.all_attention_probs_ori, \
                                     model_student_doc.all_q_w_4d, model_student_doc.all_k_w_4d

    size_per_head =  int(model_teacher.hidden_size / model_teacher.num_attention_heads)
    tf.logging.info("size_per_head: {}, expected 64 for base".format(size_per_head))

    loss, num = 0, 0

    for sbert_query_qw, sbert_doc_kw, \
        sbert_query_kw, sbert_doc_qw, \
        bert_att_score \
            in zip(stu_qu_all_q_w_4d, stu_do_all_k_w_4d,
                   stu_qu_all_k_w_4d, stu_do_all_q_w_4d,
                   tea_all_att_scores_before_mask):
        # sbert_query_qw: [bs, num_heads, seq_len=130, head_dim]
        # sbert_doc_kw: [bs, num_heads, seq_len, head_dim]
        # bert_att_score: [bs, num_heads, 2*seq_len-1, 2*seq_len-1]
        query_doc_qk = tf.matmul(sbert_query_qw[:, :, 1:-1, :], sbert_doc_kw[:, :, 1:-1, :], transpose_b=True)  #[bs, num_heads, 128, 128]
        query_doc_qk = tf.multiply(query_doc_qk,
                                 1.0 / math.sqrt(float(size_per_head)))
        query_doc_att_matrix_mask = create_att_mask(input_mask_sbert_doc)       # doc中的padding元素不应该被attend, [bs, 130, 130]
        query_doc_att_matrix_mask = tf.expand_dims(query_doc_att_matrix_mask[:, 1:-1, 1:-1], axis=[1])  #to [bs, 1, seq_len=128, seq_len=128]
        query_doc_att_matrix_mask_adder = (1.0 - tf.cast(query_doc_att_matrix_mask, tf.float32)) * -10000.0
        query_doc_att_scores = query_doc_qk + query_doc_att_matrix_mask_adder
        query_doc_att_probs = tf.nn.softmax(query_doc_att_scores)

        sbert_att_shape = modeling.get_shape_list(sbert_query_qw, expected_rank=4)  # [bs, num_heads, seq_len, head_dim]
        seq_len_sbert = sbert_att_shape[2]
        bert_att_score_query_doc = bert_att_score[:,:, 1:(seq_len_sbert-1), seq_len_sbert:-1]    #[bs, num_heads, seq_len=128, seq_len=128]
        bert_att_score_query_doc = bert_att_score_query_doc + query_doc_att_matrix_mask_adder
        bert_att_probs_query_doc = tf.nn.softmax(bert_att_score_query_doc)
        loss = loss + tf.losses.mean_squared_error(query_doc_att_probs, bert_att_probs_query_doc)
        #-------------------------------------------------------------------


        doc_query_qk = tf.matmul(sbert_doc_qw[:, :, 1:-1, :], sbert_query_kw[:, :, 1:-1, :], transpose_b=True)  #[bs, num_heads, 128, 128]
        doc_query_qk = tf.multiply(doc_query_qk,
                                1.0 / math.sqrt(float(size_per_head)))
        doc_query_att_matrix_mask = create_att_mask(input_mask_sbert_query)
        doc_query_att_matrix_mask = tf.expand_dims(doc_query_att_matrix_mask[:, 1:-1, 1:-1], axis=[1])  #to [bs, 1, seq_len=128, seq_len=128]
        doc_query_att_matrix_mask_adder = (1.0 - tf.cast(doc_query_att_matrix_mask, tf.float32)) * -10000.0
        doc_query_att_scores = doc_query_qk + doc_query_att_matrix_mask_adder
        doc_query_att_probs = tf.nn.softmax(doc_query_att_scores)

        bert_att_score_doc_query = bert_att_score[:, :, seq_len_sbert:-1, 1:(seq_len_sbert-1)]
        bert_att_score_doc_query = bert_att_score_doc_query + doc_query_att_matrix_mask_adder
        bert_att_probs_doc_query = tf.nn.softmax(bert_att_score_doc_query)
        loss = loss + tf.losses.mean_squared_error(doc_query_att_probs, bert_att_probs_doc_query)

        num += 1

    loss = loss / num

    return loss



def get_attention_loss_with_weight_layer(model_student_query, model_student_doc, model_teacher,
                       input_mask_sbert_query, input_mask_sbert_doc):
    """
        获取交互的attention loss
        """
    tea_all_att_scores_before_mask, tea_all_att_probs_ori, \
    tea_all_q_w_4d, tea_all_k_w_4d = model_teacher.all_attention_scores_before_mask, model_teacher.all_attention_probs_ori, \
                                     model_teacher.all_q_w_4d, model_teacher.all_k_w_4d

    stu_qu_all_att_scores_before_mask, stu_qu_all_att_probs_ori, \
    stu_qu_all_q_w_4d, stu_qu_all_k_w_4d = model_student_query.all_attention_scores_before_mask, model_student_query.all_attention_probs_ori, \
                                           model_student_query.all_q_w_4d, model_student_query.all_k_w_4d

    stu_do_all_att_scores_before_mask, stu_do_all_att_probs_ori, \
    stu_do_all_q_w_4d, stu_do_all_k_w_4d = model_student_doc.all_attention_scores_before_mask, model_student_doc.all_attention_probs_ori, \
                                           model_student_doc.all_q_w_4d, model_student_doc.all_k_w_4d

    size_per_head = int(model_teacher.hidden_size / model_teacher.num_attention_heads)
    tf.logging.info("size_per_head: {}, expected 64 for base".format(size_per_head))

    # loss, num = 0, 0

    flat_query2doc_list_bi,      flat_doc2query_list_bi = [], []
    flat_query2doc_list_cross,   flat_doc2query_list_cross = [], []
    origin_query2doc_list_bi,    origin_doc2query_list_bi = [], []
    origin_query2doc_list_cross, origin_doc2query_list_cross = [], []


    for sbert_query_qw, sbert_doc_kw, \
        sbert_query_kw, sbert_doc_qw, \
        bert_att_score \
            in zip(stu_qu_all_q_w_4d, stu_do_all_k_w_4d,
                   stu_qu_all_k_w_4d, stu_do_all_q_w_4d,
                   tea_all_att_scores_before_mask):
        # sbert_query_qw: [bs, num_heads, seq_len=130, head_dim]
        # sbert_doc_kw: [bs, num_heads, seq_len, head_dim]
        # bert_att_score: [bs, num_heads, 2*seq_len-1, 2*seq_len-1]
        query_doc_qk = tf.matmul(sbert_query_qw[:, :, 1:-1, :], sbert_doc_kw[:, :, 1:-1, :],
                                 transpose_b=True)  # [bs, num_heads, 128, 128]
        query_doc_qk = tf.multiply(query_doc_qk,
                                   1.0 / math.sqrt(float(size_per_head)))
        query_doc_att_matrix_mask = create_att_mask(input_mask_sbert_doc)  # doc中的padding元素不应该被attend, [bs, 130, 130]
        query_doc_att_matrix_mask = tf.expand_dims(query_doc_att_matrix_mask[:, 1:-1, 1:-1],
                                                   axis=[1])  # to [bs, 1, seq_len=128, seq_len=128]
        query_doc_att_matrix_mask_multiplyer = tf.cast(query_doc_att_matrix_mask, tf.float32)
        query_doc_att_matrix_mask_adder = (1.0 - tf.cast(query_doc_att_matrix_mask, tf.float32)) * -10000.0
        query_doc_att_scores = query_doc_qk + query_doc_att_matrix_mask_adder
        query_doc_att_probs = tf.nn.softmax(query_doc_att_scores)
        origin_query2doc_list_bi.append(query_doc_att_probs)            # ************************
        #------------------------newly added for computed weighted mapping--------------------------------------------
        multiplyed_query2doc_bi = query_doc_qk * query_doc_att_matrix_mask_multiplyer  # [bs, num_heads, seq_len, seq_len]
        flat_query2doc_bi = tf.reduce_mean(multiplyed_query2doc_bi, axis=-1)     #[bs, num_heads, seq_len]
        flat_query2doc_bi = tf.reduce_mean(flat_query2doc_bi, axis=1)       # [bs, seq_len]
        flat_query2doc_list_bi.append(flat_query2doc_bi)
        #----------------------------------------------------------------------------------------------------------------
        sbert_att_shape = modeling.get_shape_list(sbert_query_qw, expected_rank=4)  # [bs, num_heads, seq_len, head_dim]
        seq_len_sbert = sbert_att_shape[2]
        bert_att_score_query_doc = bert_att_score[:, :, 1:(seq_len_sbert - 1),
                                   seq_len_sbert:-1]  # [bs, num_heads, seq_len=128, seq_len=128]
        multiplyed_query2doc_cross = bert_att_score_query_doc * query_doc_att_matrix_mask_multiplyer
        bert_att_score_query_doc = bert_att_score_query_doc + query_doc_att_matrix_mask_adder
        bert_att_probs_query_doc = tf.nn.softmax(bert_att_score_query_doc)
        origin_query2doc_list_cross.append(bert_att_probs_query_doc)        # ************************
        #------------------------------newly added for computed weighted mapping--------------------------------------
        flat_query2doc_cross = tf.reduce_mean(multiplyed_query2doc_cross, axis=-1)
        flat_query2doc_cross = tf.reduce_mean(flat_query2doc_cross, axis=1)
        flat_query2doc_list_cross.append(flat_query2doc_cross)


        # loss = loss + tf.losses.mean_squared_error(query_doc_att_probs, bert_att_probs_query_doc)
        # -------------------------------------------------------------------


        doc_query_qk = tf.matmul(sbert_doc_qw[:, :, 1:-1, :], sbert_query_kw[:, :, 1:-1, :],
                                 transpose_b=True)  # [bs, num_heads, 128, 128]
        doc_query_qk = tf.multiply(doc_query_qk,
                                   1.0 / math.sqrt(float(size_per_head)))
        doc_query_att_matrix_mask = create_att_mask(input_mask_sbert_query)
        doc_query_att_matrix_mask = tf.expand_dims(doc_query_att_matrix_mask[:, 1:-1, 1:-1],
                                                   axis=[1])  # to [bs, 1, seq_len=128, seq_len=128]
        doc_query_att_matrix_mask_adder = (1.0 - tf.cast(doc_query_att_matrix_mask, tf.float32)) * -10000.0
        doc_query_att_matrix_mask_multiplyer = tf.cast(doc_query_att_matrix_mask, tf.float32)
        doc_query_att_scores = doc_query_qk + doc_query_att_matrix_mask_adder
        doc_query_att_probs = tf.nn.softmax(doc_query_att_scores)
        origin_doc2query_list_bi.append(doc_query_att_probs)            # ************************
        # ------------------------newly added for computed weighted mapping--------------------------------------------

        multiplyed_doc2query_bi = doc_query_qk * doc_query_att_matrix_mask_multiplyer  # [bs, num_heads, seq_len, seq_len]
        flat_doc2query_bi = tf.reduce_mean(multiplyed_doc2query_bi, axis=-1)  # [bs, num_heads, seq_len]
        flat_doc2query_bi = tf.reduce_mean(flat_doc2query_bi, axis=1)
        flat_doc2query_list_bi.append(flat_doc2query_bi)
        # ----------------------------------------------------------------------------------------------------------------

        bert_att_score_doc_query = bert_att_score[:, :, seq_len_sbert:-1, 1:(seq_len_sbert - 1)]
        #------------------------newly added for computed weighted mapping--------------------------------------------
        multiplyed_doc2query_cross = bert_att_score_doc_query * doc_query_att_matrix_mask_multiplyer
        flat_doc2query_cross = tf.reduce_mean(multiplyed_doc2query_cross, axis=-1)
        flat_doc2query_cross = tf.reduce_mean(flat_doc2query_cross, axis=1)
        flat_doc2query_list_cross.append(flat_doc2query_cross)

        bert_att_score_doc_query = bert_att_score_doc_query + doc_query_att_matrix_mask_adder
        bert_att_probs_doc_query = tf.nn.softmax(bert_att_score_doc_query)
        origin_doc2query_list_cross.append(bert_att_probs_doc_query)   #************************
        # loss = loss + tf.losses.mean_squared_error(doc_query_att_probs, bert_att_probs_doc_query)

        # num += 1

    flat_att_bi = []
    for query2doc_bi, doc2query_bi in zip(flat_query2doc_list_bi, flat_doc2query_list_bi):
        # [bs, seq_len1], [bs, seq_len2]
        layer_att_bi = tf.concat([query2doc_bi, doc2query_bi], axis=-1)
        flat_att_bi.append(layer_att_bi)

    flat_att_cross = []
    for query2doc_cross, doc2query_cross in zip(flat_query2doc_list_cross, flat_doc2query_list_cross):
        # [bs, seq_len1], [bs, seq_len2]
        print("******************************, shape flat_att_cross_query2doc:", modeling.get_shape_list(query2doc_cross))
        print("******************************, shape flat_att_cross_doc2query:", modeling.get_shape_list(doc2query_cross))
        layer_att_cross = tf.concat([query2doc_cross, doc2query_cross], axis=-1)
        flat_att_cross.append(layer_att_cross)

    flat_att_bi = tf.stack(flat_att_bi, axis=0)                 #[12, bs, seq_len(dim)]
    flat_att_cross = tf.stack(flat_att_cross, axis=0)           #[12, bs, seq_len(dim)]
    print("******************************, shape flat_att_bi:", modeling.get_shape_list(flat_att_bi))
    print("******************************, shape flat_att_cross:", modeling.get_shape_list(flat_att_cross))
    flat_att_bi = tf.transpose(flat_att_bi, [1, 0, 2])          #[bs, 12, seq_len(dim)]
    flat_att_cross = tf.transpose(flat_att_cross, [1, 0, 2])    #[bs, 12, seq_len(dim)]
    dim = modeling.get_shape_list(flat_att_bi, expected_rank=[3])[-1]
    mapped_flat_att_bi = tf.layers.dense(
        flat_att_bi,
        dim,
        activation=None,
        name="bi2cross_map",
        kernel_initializer=modeling.create_initializer(0.02)
    )       #[bs, 12, seq_len(dim)]
    mapped_score = tf.matmul(mapped_flat_att_bi, flat_att_cross, transpose_b=True)      # [bs, 12, 12]
    mapped_score = tf.multiply(mapped_score,
                                   1.0 / math.sqrt(float(dim)))
    mapped_score = tf.nn.softmax(mapped_score)

    origin_query2doc_list_cross = tf.stack(origin_query2doc_list_cross, axis=0)  #[12, bs, num_heads, seq_len, seq_len]
    origin_query2doc_list_cross = tf.transpose(origin_query2doc_list_cross, [1, 0, 2, 3, 4])    #[bs, 12, num_heads, seq_len1, seq_len2]
    bs, layer_num, num_heads, seq_l1, seq_l2 = modeling.get_shape_list(origin_query2doc_list_cross, expected_rank=[5])
    origin_query2doc_list_cross = tf.reshape(origin_query2doc_list_cross, [bs, layer_num, -1])  # [bs, 12, num_heads*seq_len1*seq_len2]
    weighted_query2doc_list_cross = tf.matmul(mapped_score, origin_query2doc_list_cross)    # [bs, 12, num_heads*seq_len1*seq_len2]
    weighted_query2doc_list_cross = tf.reshape(weighted_query2doc_list_cross, [layer_num,bs,num_heads,seq_l1,seq_l2])

    origin_doc2query_list_cross = tf.stack(origin_doc2query_list_cross, axis=0) #[12, bs, num_heads, seq_len, seq_len]
    origin_doc2query_list_cross = tf.transpose(origin_doc2query_list_cross, [1, 0, 2, 3, 4])    #[bs, 12, num_heads, seq_len1, seq_len2]
    _1bs, _1layer_num, _1num_heads, _1seq_l1, _1seq_l2 = modeling.get_shape_list(origin_doc2query_list_cross, expected_rank=[5])
    origin_doc2query_list_cross = tf.reshape(origin_doc2query_list_cross, [_1bs, _1layer_num, -1])  # [bs, 12, num_heads*seq_len1*seq_len2]
    weighted_doc2query_list_cross = tf.matmul(mapped_score, origin_doc2query_list_cross)
    weighted_doc2query_list_cross = tf.reshape(weighted_doc2query_list_cross, [_1layer_num,_1bs,_1num_heads,_1seq_l1,_1seq_l2])

    origin_query2doc_list_bi = tf.stack(origin_query2doc_list_bi, axis=0)   #[12, bs, num_heads, seq_len, seq_len]
    origin_doc2query_list_bi = tf.stack(origin_doc2query_list_bi, axis=0)   #[12, bs, num_heads, seq_len, seq_len]

    loss_query2doc = tf.square(origin_query2doc_list_bi - weighted_query2doc_list_cross)
    loss_query2doc = tf.reduce_sum(loss_query2doc) / (1.0 * tf.cast(bs * num_heads * layer_num * seq_l1, dtype=tf.float32))

    loss_doc2query = tf.square(origin_doc2query_list_bi - weighted_doc2query_list_cross)
    loss_doc2query = tf.reduce_sum(loss_doc2query) / (1.0 * tf.cast(_1bs * _1num_heads * _1layer_num * _1seq_l1, dtype=tf.float32))

    loss = (loss_query2doc + loss_doc2query) / 2.0

    return loss


def get_attention_loss_with_weight_head(model_student_query, model_student_doc, model_teacher,
                       input_mask_sbert_query, input_mask_sbert_doc):
    """
        获取交互的attention loss
        """
    tea_all_att_scores_before_mask, tea_all_att_probs_ori, \
    tea_all_q_w_4d, tea_all_k_w_4d = model_teacher.all_attention_scores_before_mask, model_teacher.all_attention_probs_ori, \
                                     model_teacher.all_q_w_4d, model_teacher.all_k_w_4d

    stu_qu_all_att_scores_before_mask, stu_qu_all_att_probs_ori, \
    stu_qu_all_q_w_4d, stu_qu_all_k_w_4d = model_student_query.all_attention_scores_before_mask, model_student_query.all_attention_probs_ori, \
                                           model_student_query.all_q_w_4d, model_student_query.all_k_w_4d

    stu_do_all_att_scores_before_mask, stu_do_all_att_probs_ori, \
    stu_do_all_q_w_4d, stu_do_all_k_w_4d = model_student_doc.all_attention_scores_before_mask, model_student_doc.all_attention_probs_ori, \
                                           model_student_doc.all_q_w_4d, model_student_doc.all_k_w_4d

    size_per_head = int(model_teacher.hidden_size / model_teacher.num_attention_heads)
    tf.logging.info("size_per_head: {}, expected 64 for base".format(size_per_head))

    # loss, num = 0, 0

    flat_query2doc_list_bi,      flat_doc2query_list_bi = [], []
    flat_query2doc_list_cross,   flat_doc2query_list_cross = [], []
    origin_query2doc_list_bi,    origin_doc2query_list_bi = [], []
    origin_query2doc_list_cross, origin_doc2query_list_cross = [], []


    for sbert_query_qw, sbert_doc_kw, \
        sbert_query_kw, sbert_doc_qw, \
        bert_att_score \
            in zip(stu_qu_all_q_w_4d, stu_do_all_k_w_4d,
                   stu_qu_all_k_w_4d, stu_do_all_q_w_4d,
                   tea_all_att_scores_before_mask):
        # sbert_query_qw: [bs, num_heads, seq_len=130, head_dim]
        # sbert_doc_kw: [bs, num_heads, seq_len, head_dim]
        # bert_att_score: [bs, num_heads, 2*seq_len-1, 2*seq_len-1]
        query_doc_qk = tf.matmul(sbert_query_qw[:, :, 1:-1, :], sbert_doc_kw[:, :, 1:-1, :],
                                 transpose_b=True)  # [bs, num_heads, 128, 128]
        query_doc_qk = tf.multiply(query_doc_qk,
                                   1.0 / math.sqrt(float(size_per_head)))
        query_doc_att_matrix_mask = create_att_mask(input_mask_sbert_doc)  # doc中的padding元素不应该被attend, [bs, 130, 130]
        query_doc_att_matrix_mask = tf.expand_dims(query_doc_att_matrix_mask[:, 1:-1, 1:-1],
                                                   axis=[1])  # to [bs, 1, seq_len=128, seq_len=128]
        query_doc_att_matrix_mask_multiplyer = tf.cast(query_doc_att_matrix_mask, tf.float32)
        query_doc_att_matrix_mask_adder = (1.0 - tf.cast(query_doc_att_matrix_mask, tf.float32)) * -10000.0
        query_doc_att_scores = query_doc_qk + query_doc_att_matrix_mask_adder
        query_doc_att_probs = tf.nn.softmax(query_doc_att_scores)
        origin_query2doc_list_bi.append(query_doc_att_probs)            # ************************
        #------------------------newly added for computed weighted mapping--------------------------------------------
        multiplyed_query2doc_bi = query_doc_qk * query_doc_att_matrix_mask_multiplyer  # [bs, num_heads, seq_len, seq_len]
        flat_query2doc_bi = tf.reduce_mean(multiplyed_query2doc_bi, axis=-1)     #[bs, num_heads, seq_len]
        # flat_query2doc_bi = tf.reduce_mean(flat_query2doc_bi, axis=1)       # [bs, seq_len]
        flat_query2doc_list_bi.append(flat_query2doc_bi)
        #----------------------------------------------------------------------------------------------------------------
        sbert_att_shape = modeling.get_shape_list(sbert_query_qw, expected_rank=4)  # [bs, num_heads, seq_len, head_dim]
        seq_len_sbert = sbert_att_shape[2]
        bert_att_score_query_doc = bert_att_score[:, :, 1:(seq_len_sbert - 1),
                                   seq_len_sbert:-1]  # [bs, num_heads, seq_len=128, seq_len=128]
        multiplyed_query2doc_cross = bert_att_score_query_doc * query_doc_att_matrix_mask_multiplyer
        bert_att_score_query_doc = bert_att_score_query_doc + query_doc_att_matrix_mask_adder
        bert_att_probs_query_doc = tf.nn.softmax(bert_att_score_query_doc)
        origin_query2doc_list_cross.append(bert_att_probs_query_doc)        # ************************
        #------------------------------newly added for computed weighted mapping--------------------------------------
        flat_query2doc_cross = tf.reduce_mean(multiplyed_query2doc_cross, axis=-1)  # [bs, num_heads, seq_len]
        # flat_query2doc_cross = tf.reduce_mean(flat_query2doc_cross, axis=1)
        flat_query2doc_list_cross.append(flat_query2doc_cross)


        # loss = loss + tf.losses.mean_squared_error(query_doc_att_probs, bert_att_probs_query_doc)
        # -------------------------------------------------------------------


        doc_query_qk = tf.matmul(sbert_doc_qw[:, :, 1:-1, :], sbert_query_kw[:, :, 1:-1, :],
                                 transpose_b=True)  # [bs, num_heads, 128, 128]
        doc_query_qk = tf.multiply(doc_query_qk,
                                   1.0 / math.sqrt(float(size_per_head)))
        doc_query_att_matrix_mask = create_att_mask(input_mask_sbert_query)
        doc_query_att_matrix_mask = tf.expand_dims(doc_query_att_matrix_mask[:, 1:-1, 1:-1],
                                                   axis=[1])  # to [bs, 1, seq_len=128, seq_len=128]
        doc_query_att_matrix_mask_adder = (1.0 - tf.cast(doc_query_att_matrix_mask, tf.float32)) * -10000.0
        doc_query_att_matrix_mask_multiplyer = tf.cast(doc_query_att_matrix_mask, tf.float32)
        doc_query_att_scores = doc_query_qk + doc_query_att_matrix_mask_adder
        doc_query_att_probs = tf.nn.softmax(doc_query_att_scores)
        origin_doc2query_list_bi.append(doc_query_att_probs)            # ************************
        # ------------------------newly added for computed weighted mapping--------------------------------------------

        multiplyed_doc2query_bi = doc_query_qk * doc_query_att_matrix_mask_multiplyer  # [bs, num_heads, seq_len, seq_len]
        flat_doc2query_bi = tf.reduce_mean(multiplyed_doc2query_bi, axis=-1)  # [bs, num_heads, seq_len]
        # flat_doc2query_bi = tf.reduce_mean(flat_doc2query_bi, axis=1)
        flat_doc2query_list_bi.append(flat_doc2query_bi)
        # ----------------------------------------------------------------------------------------------------------------

        bert_att_score_doc_query = bert_att_score[:, :, seq_len_sbert:-1, 1:(seq_len_sbert - 1)]
        #------------------------newly added for computed weighted mapping--------------------------------------------
        multiplyed_doc2query_cross = bert_att_score_doc_query * doc_query_att_matrix_mask_multiplyer
        flat_doc2query_cross = tf.reduce_mean(multiplyed_doc2query_cross, axis=-1)
        # flat_doc2query_cross = tf.reduce_mean(flat_doc2query_cross, axis=1)
        flat_doc2query_list_cross.append(flat_doc2query_cross)

        bert_att_score_doc_query = bert_att_score_doc_query + doc_query_att_matrix_mask_adder
        bert_att_probs_doc_query = tf.nn.softmax(bert_att_score_doc_query)
        origin_doc2query_list_cross.append(bert_att_probs_doc_query)   #************************
        # loss = loss + tf.losses.mean_squared_error(doc_query_att_probs, bert_att_probs_doc_query)

        # num += 1

    flat_att_bi = []
    for query2doc_bi, doc2query_bi in zip(flat_query2doc_list_bi, flat_doc2query_list_bi):
        # [bs, num_heads, seq_len1], [bs, num_heads, seq_len2]
        layer_att_bi = tf.concat([query2doc_bi, doc2query_bi], axis=-1)
        flat_att_bi.append(layer_att_bi)

    flat_att_cross = []
    for query2doc_cross, doc2query_cross in zip(flat_query2doc_list_cross, flat_doc2query_list_cross):
        # [bs, num_heads, seq_len1], [bs, num_heads, seq_len2]
        print("******************************, shape flat_att_cross_query2doc:", modeling.get_shape_list(query2doc_cross))
        print("******************************, shape flat_att_cross_doc2query:", modeling.get_shape_list(doc2query_cross))
        layer_att_cross = tf.concat([query2doc_cross, doc2query_cross], axis=-1)        #[bs, num_heads, len1+len2]
        flat_att_cross.append(layer_att_cross)

    # flat_att_bi = tf.stack(flat_att_bi, axis=0)                 #[12, bs, num_heads, seq_len(dim)]
    # flat_att_cross = tf.stack(flat_att_cross, axis=0)           #[12, bs, num_heads, seq_len(dim)]
    dim = modeling.get_shape_list(flat_att_bi[-1], expected_rank=[3])[-1]
    loss, cnt = 0, 0
    for layer_id, (layer_att_bi, layer_att_cross, layer_query2doc_bi, layer_query2doc_cross, layer_doc2query_bi, layer_doc2query_cross) in \
                                                                    enumerate(zip(flat_att_bi, flat_att_cross,
                                                                   origin_query2doc_list_bi, origin_query2doc_list_cross,
                                                                   origin_doc2query_list_bi, origin_doc2query_list_cross)):
        mapped_flat_att_bi = tf.layers.dense(
            layer_att_bi,
            dim,
            activation=None,
            name="bi2cross_map_"+str(layer_id),
            kernel_initializer=modeling.create_initializer(0.02)
        )  # [bs, num_heads, seq_len(dim)]
        mapped_score = tf.matmul(mapped_flat_att_bi, layer_att_cross, transpose_b=True) #[bs, num_heads, num_heads]
        mapped_score = tf.multiply(mapped_score,
                                   1.0 / math.sqrt(float(dim)))
        mapped_score = tf.nn.softmax(mapped_score)                  #[bs, num_heads, num_heads]
        # layer_query2doc_cross: [bs, num_heads, seq_len, seq_len]
        bs, num_heads, seq_len1_query2doc, seq_len2_query2doc = modeling.get_shape_list(layer_query2doc_cross, expected_rank=[4])
        _layer_query2doc_cross = tf.reshape(layer_query2doc_cross, [bs, num_heads, -1])
        weighted_layer_query2doc_cross = tf.matmul(mapped_score, _layer_query2doc_cross)
        weighted_layer_query2doc_cross = tf.reshape(weighted_layer_query2doc_cross, [bs, num_heads, seq_len1_query2doc, seq_len2_query2doc])
        # [bs, num_heads, seq_len, seq_len]
        loss_query2doc = tf.square(weighted_layer_query2doc_cross - layer_query2doc_bi)
        loss_query2doc = tf.reduce_sum(loss_query2doc) / (
                    1.0 * tf.cast(bs * num_heads * seq_len1_query2doc, dtype=tf.float32))

        bs, num_heads, seq_len1_doc2query, seq_len2_doc2query = modeling.get_shape_list(layer_doc2query_cross,
                                                                                        expected_rank=[4])
        _layer_doc2query_cross = tf.reshape(layer_doc2query_cross, [bs, num_heads, -1])
        weighted_layer_doc2query_cross = tf.matmul(mapped_score, _layer_doc2query_cross)
        weighted_layer_doc2query_cross = tf.reshape(weighted_layer_doc2query_cross, [bs, num_heads, seq_len1_doc2query, seq_len2_doc2query])
        loss_doc2query = tf.square(weighted_layer_doc2query_cross - layer_doc2query_bi)
        loss_doc2query = tf.reduce_sum(loss_doc2query) / (
                    1.0 * tf.cast(bs * num_heads * seq_len1_doc2query, dtype=tf.float32))

        loss += (loss_query2doc+loss_doc2query)
        cnt += 1.0

    return loss / cnt


def get_pooled_embeddings(encode_layer, input_mask):
    """
    获取mean pool向量，同时去除input中padding项(mask为0)的影响。
    encoder_layer:  [bs, seq_len, emb_dim]
    input_mask: [bs, seq_len]
    """
    mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
    masked_output_layer = mask * encode_layer  # [bs_size, max_len, emb_dim]
    sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
    actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
    actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
    output_layer = sum_masked_output_layer / actual_token_nums
    return output_layer     # [bs_size, emb_dim]


def get_pooled_embeddings_for2(encoder_layer1, encoder_layer2, input_mask1, input_mask2):
    """
    获取两个layer整体的mean pool向量，同时去除input中padding项(mask为0)的影响。
    encoder_layer:  [bs, seq_len, emb_dim]
    input_mask: [bs, seq_len]
    """
    input_mask = tf.concat([input_mask1, input_mask2], axis=-1)     #[bs, seq_len1+seq_len2]
    encoder_layer = tf.concat([encoder_layer1, encoder_layer2], axis=1)   #[bs, seq_len1+seq_len2, emb_dim]
    output_layer = get_pooled_embeddings(encoder_layer, input_mask)
    return output_layer



def use_all_layer_embedding(stu_model_query, stu_model_doc, input_mask_query, input_mask_doc):
    all_encoder_layers_query = stu_model_query.all_encoder_layers   #[
    pre_encoder_layers_query = all_encoder_layers_query[:-1]
    last_encoder_layer_query = all_encoder_layers_query[-1]
    #-----------------------------------------------------------
    all_encoder_layers_doc = stu_model_doc.all_encoder_layers
    pre_encoder_layers_doc = all_encoder_layers_doc[:-1]
    last_encoder_layer_doc = all_encoder_layers_doc[-1]
    #------------------ perform pooling for each sentence-------------------
    pooled_pre_layers_query = []
    for pre_layer_query in pre_encoder_layers_query:        # [bs, seq_len, emb_dim]
        pooled_layer = get_pooled_embeddings(pre_layer_query, input_mask_query)
        pooled_pre_layers_query.append(pooled_layer)
    pooled_pre_layers_doc = []
    for pre_layer_doc in pre_encoder_layers_doc:
        pooled_layer = get_pooled_embeddings(pre_layer_doc, input_mask_doc)
        pooled_pre_layers_doc.append(pooled_layer)
    #----------------- perform <max(a,b), |a-b|, a, b>---------------------------
    combine_pre_layers = []
    for layer_query, layer_doc in zip(pooled_pre_layers_query, pooled_pre_layers_doc):
        sub_embedding = tf.abs(layer_query - layer_doc)
        max_embedding = tf.square(tf.reduce_max([layer_query, layer_doc], axis=0))
        regular_embedding = tf.concat([layer_query, layer_doc, sub_embedding, max_embedding], -1)
        combine_pre_layers.append(regular_embedding)
    #------------------ perform weighted aggregate--------------------------
    pre_layer_num = len(combine_pre_layers)
    tf.logging.info('********pre layer nums: {}***********'.format(pre_layer_num))
    weight = tf.get_variable("aggregate_weight", [pre_layer_num],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
    combine_pre_layers = tf.stack(combine_pre_layers, axis=0)       # [pre_layer_num, bs, emb_dim*4]
    weight_1 = tf.expand_dims(tf.expand_dims(weight,axis=-1), axis=-1)      # [pre_layer_num, 1, 1]
    weighted_multiply = combine_pre_layers * weight_1
    weighted_sum = tf.reduce_sum(weighted_multiply, axis=0)             #[bs, emb_dim*4]
    #-------------------perform last layer-------------------------------------
    last_pooled_layer_query = get_pooled_embeddings(last_encoder_layer_query, input_mask_query)
    last_pooled_layer_doc = get_pooled_embeddings(last_encoder_layer_doc, input_mask_doc)
    last_sub_embedding = tf.abs(last_pooled_layer_query - last_pooled_layer_doc)
    last_max_embedding = tf.square(tf.reduce_max([last_pooled_layer_query, last_pooled_layer_doc], axis=0))
    last_regular_embedding = tf.concat([last_pooled_layer_query, last_pooled_layer_doc, last_sub_embedding, last_max_embedding], -1)

    #--------------combine weighted_sum and last layer-----------------------
    final_embedding = tf.reduce_max([weighted_sum, last_regular_embedding], axis=0)

    return final_embedding







def cosine_distance(X1, X2):
    """
    余弦相似度
    X1 : [bs, emb_dim]
    X2:  [bs, emb_dim]
    """
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=-1))    #[bs]
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))    #[bs]
    X1_norm_X2_norm = X1_norm * X2_norm

    X1_X2 = X1 * X2     #[bs * emb_dim]
    X1_X2 = tf.reduce_sum(X1_X2, axis=-1)           #[bs]

    cosine = X1_X2 / X1_norm_X2_norm        # 相似度，越大越好, [bs]
    cosine = tf.reduce_mean(cosine, axis=-1)
    return 1-cosine             #distance


def get_pooled_loss(teacher_model, student_model_query, student_model_doc,
                    input_mask_teacher, input_mask_query, input_mask_doc,
                    mode):
    """
    计算pooled的representation之间的差异；其中：
    teacher采用mean pooling
    mode1: [direct_mean] student采用整体的mean pooling，此时和teacher的mean pooling为同一维度
    mode2: [map_mean] student分别将query和doc进行mean pooling, 然后调用|v1-v2, v1+v2...|，然后映射到teacher 维度
    """
    all_teacher_layers, all_query_layers, all_doc_layers = \
        teacher_model.all_encoder_layers, \
        student_model_query.all_encoder_layers, \
        student_model_doc.all_encoder_layers
    if mode == "direct_mean":
        tf.logging.info('*****use direct mean as hidden pooling...')
        loss, cnt = 0, 0
        for teacher_layer, query_layer, doc_layer in zip(all_teacher_layers, all_query_layers, all_doc_layers):
            # each layer is [bs, seq_len, emb_dim]
            pooled_teacher_layer = get_pooled_embeddings(teacher_layer, input_mask_teacher)     #[bs, emb_dim]
            pooled_student_layer = get_pooled_embeddings_for2(query_layer, doc_layer, input_mask_query, input_mask_doc)
            # loss += tf.reduce_sum(tf.square(pooled_teacher_layer-pooled_student_layer))     # squared error
            loss += cosine_distance(pooled_student_layer, pooled_teacher_layer)
            cnt = 1
        return loss / cnt
    elif mode == "map_mean":
        tf.logging.info('*****use map mean as hidden pooling...')
        loss, cnt = 0, 0
        map_weights = tf.get_variable(
            "map_weights", [teacher_model.hidden_size * 4, teacher_model.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        map_bias = tf.get_variable(
            "map_bias", [teacher_model.hidden_size],
            initializer=tf.zeros_initializer())
        for teacher_layer, query_layer, doc_layer in zip(all_teacher_layers, all_query_layers, all_doc_layers):
            # each layer is [bs, seq_len, emb_dim]
            pooled_teacher_layer = get_pooled_embeddings(teacher_layer, input_mask_teacher)     #[bs, emb_dim]
            pooled_query_layer = get_pooled_embeddings(query_layer, input_mask_query)       #[bs, emb_dim]
            pooled_doc_layer = get_pooled_embeddings(doc_layer, input_mask_doc)             #[bs, emb_dim]
            sub_embedding = tf.abs(pooled_query_layer - pooled_doc_layer)
            max_embedding = tf.square(tf.reduce_max([pooled_query_layer, pooled_doc_layer], axis=0))
            regular_embedding = tf.concat([pooled_query_layer, pooled_doc_layer, sub_embedding, max_embedding], -1)

            mapped_student_layer = tf.matmul(regular_embedding, map_weights)
            mapped_student_layer = tf.nn.bias_add(mapped_student_layer, map_bias)

            # loss += tf.reduce_sum(tf.square(pooled_teacher_layer-mapped_student_layer))     # squared error
            loss += cosine_distance(pooled_teacher_layer, mapped_student_layer)
            cnt = 1
        return loss / cnt
    else:
        assert 1==2


def cos_sim_loss_for_contrast(matrix_a, matrix_b):
    """
    matrix_a: batch_size * emb_dim
    matrix_b: batch_size * emb_dim
    return: cos_sim contrastive loss
    """
    dot_result = tf.matmul(matrix_a, matrix_b, transpose_b=True)
    norm2_a_output = tf.sqrt(tf.reduce_sum(tf.square(matrix_a), axis=1, keep_dims=True))
    norm2_b_output = tf.sqrt(tf.reduce_sum(tf.square(matrix_b), axis=1, keep_dims=True))
    norm2_ab = tf.matmul(norm2_a_output, norm2_b_output, transpose_b=True)
    cos_sim = tf.divide(dot_result, norm2_ab)  # batch_size * batch_size
    cos_sim = tf.nn.softmax(cos_sim, axis=1)
    log_cos_sim = tf.log(cos_sim)
    diag_elem = tf.multiply(tf.eye(tf.shape(cos_sim)[0]), log_cos_sim)
    diag_sum = tf.reduce_sum(diag_elem)
    return diag_sum


def contrastive_loss(teacher_model, query_model, doc_model, regular_embedding,
                     input_mask_teacher, input_mask_query, input_mask_doc, mode):
    teacher_outputs = teacher_model.get_sequence_output()
    query_output = query_model.get_sequence_output()
    doc_output = doc_model.get_sequence_output()
    if mode == "direct_mean":
        tf.logging.info('*****use direct mean as hidden pooling...')
        loss = 0
        pooled_teacher_layer = get_pooled_embeddings(teacher_outputs, input_mask_teacher)  # [bs, emb_dim]
        pooled_student_layer = get_pooled_embeddings_for2(query_output, doc_output, input_mask_query, input_mask_doc)
        # loss += tf.reduce_sum(tf.square(pooled_teacher_layer-pooled_student_layer))     # squared error
        loss += cos_sim_loss_for_contrast(pooled_teacher_layer, pooled_student_layer)
        return loss
    elif mode == "map_mean":
        tf.logging.info('*****use map mean as hidden pooling...')
        loss = 0
        pooled_teacher_layer = get_pooled_embeddings(teacher_outputs, input_mask_teacher)  # [bs, emb_dim]
        map_weights = tf.get_variable(
            "map_weights_cts", [teacher_model.hidden_size * 4, teacher_model.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        map_bias = tf.get_variable(
            "map_bias_cts", [teacher_model.hidden_size],
            initializer=tf.zeros_initializer())
        mapped_student_layer = tf.matmul(regular_embedding, map_weights)
        mapped_student_layer = tf.nn.bias_add(mapped_student_layer, map_bias)
        loss += cos_sim_loss_for_contrast(pooled_teacher_layer, mapped_student_layer)
        return loss



def contrastive_loss_self(teacher_model, query_model, doc_model,
                          input_mask_teacher, input_mask_query, input_mask_doc,
                          truth_labels):
    """
    在双塔内部进行对比学习，即对于输入的文本对 S1 S2，
    如果S1与S2的标签是吻合，就将其作为正例；并将同batch内对面的所有其他输入作为负例；
    暂时只计算一个塔
    """
    def cos_loss_self(matrix_a, matrix_b, label_mask):
        """
        label_mask: 代表哪个是正例，即语义是吻合的, label为1  [bs]
        """
        dot_result = tf.matmul(matrix_a, matrix_b, transpose_b=True)            #[bs , bs]   dot
        norm2_a_output = tf.sqrt(tf.reduce_sum(tf.square(matrix_a), axis=1, keep_dims=True))    #[bs, 1]  |a|
        norm2_b_output = tf.sqrt(tf.reduce_sum(tf.square(matrix_b), axis=1, keep_dims=True))    #[bs, 1], |b|
        norm2_ab = tf.matmul(norm2_a_output, norm2_b_output, transpose_b=True)                  #[bs, bs], |a||b|
        cos_sim = tf.divide(dot_result, norm2_ab)  # batch_size * batch_size
        cos_sim = tf.nn.softmax(cos_sim, axis=1)
        log_cos_sim = tf.log(cos_sim)
        diag_elem = tf.multiply(tf.eye(tf.shape(cos_sim)[0]), log_cos_sim)      #[bs, bs] only diag remained
        label_mask = tf.cast(tf.expand_dims(label_mask, axis=-1), dtype=tf.float32)     #[bs, 1]
        matched_diag_elem = tf.multiply(diag_elem, label_mask)
        loss = tf.reduce_sum(matched_diag_elem)
        return loss

    all_teacher_layers, all_query_layers, all_doc_layers = \
        teacher_model.all_encoder_layers, \
        query_model.all_encoder_layers, \
        doc_model.all_encoder_layers
    loss, cnt = 0, 0
    for teacher_layer, query_layer, doc_layer in zip(all_teacher_layers, all_query_layers, all_doc_layers):
        pooled_query_layer = get_pooled_embeddings(query_layer, input_mask_query)  # [bs, emb_dim]
        pooled_doc_layer = get_pooled_embeddings(doc_layer, input_mask_doc)  # [bs, emb_dim]
        loss += cos_loss_self(pooled_query_layer, pooled_doc_layer, truth_labels)
        cnt += 1

    loss = loss / cnt

    return -1 * loss


def contrastive_loss_teacher_separately(teacher_model, query_model, doc_model,
                          input_mask_teacher, input_mask_query, input_mask_doc,
                          truth_labels):
    """
    teacher半段以query为正例，以同batch其他query内为负例;  后半段类似
        q1, q2, q3
    t1  *
    t2      *
    t3          *
    """
    all_teacher_layers, all_query_layers, all_doc_layers = \
        teacher_model.all_encoder_layers, \
        query_model.all_encoder_layers, \
        doc_model.all_encoder_layers
    loss, cnt = 0, 0
    for teacher_layer, query_layer, doc_layer in zip(all_teacher_layers, all_query_layers, all_doc_layers):
        pooled_query_layer = get_pooled_embeddings(query_layer[:, 1:-1, :], input_mask_query[:, 1:-1])  # [bs, emb_dim]
        pooled_doc_layer = get_pooled_embeddings(doc_layer[:, 1:-1, :], input_mask_doc[:, 1:-1])  # [bs, emb_dim]

        query_length = modeling.get_shape_list(query_layer, expected_rank=3)[1]   #[bs, query_len, emb_dim]
        # doc_length = modeling.get_shape_list(doc_layer, expected_rank=3)[1]
        teacher_query = get_pooled_embeddings(teacher_layer[:, 1:query_length-1, :], input_mask_teacher[:, 1:query_length-1])
        teacher_doc = get_pooled_embeddings(teacher_layer[:, query_length:-1, :], input_mask_teacher[:, query_length:-1])

        loss_query = cos_sim_loss_for_contrast(teacher_query, pooled_query_layer)
        loss_doc = cos_sim_loss_for_contrast(teacher_doc, pooled_doc_layer)
        loss += (loss_query + loss_doc) / 2.0
        cnt += 1
    return -1 * loss / cnt


def kl(p, q):
    """
    计算kl散度
    """
    p_q = p / q



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
        "mnli": MnliProcessor,
        "qqp": QqpProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_save:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length_bert > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

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
        keep_checkpoint_max = 80,
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
        init_checkpoint_teacher=FLAGS.init_checkpoint_teacher,
        init_checkpoint_student=FLAGS.init_checkpoint_student,
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
                train_examples, label_list, FLAGS.max_seq_length_bert, FLAGS.max_seq_length_sbert,
                tokenizer, train_file, is_training=True)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length_bert=FLAGS.max_seq_length_bert,
            seq_length_sbert=FLAGS.max_seq_length_sbert,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length_bert, FLAGS.max_seq_length_sbert,
                tokenizer, eval_file)
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
            seq_length_bert=FLAGS.max_seq_length_bert,
            seq_length_sbert=FLAGS.max_seq_length_sbert,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        steps_and_files = []
        filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
        for file in filenames:
            if file.endswith(".index"):
                ckpt_name = file[:-6]
                cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                steps_and_files.append((global_step, cur_filename))
                tf.logging.info("add {} to eval list...".format(cur_filename))

        steps_and_files = sorted(steps_and_files, key=lambda x:x[0])

        best_metric, best_ckpt = 0, ''
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for global_step, filename in steps_and_files:
                result = estimator.evaluate(input_fn=eval_input_fn,
                                            # steps=eval_steps)
                                            checkpoint_path=filename)
                cur_acc = result["eval_accuracy"]
                if cur_acc > best_metric:
                    best_metric = cur_acc
                    best_ckpt = filename
                tf.logging.info("***** Eval results of step-{} *****".format(global_step))
                writer.write("***** Eval results of step-{} *****".format(global_step))
                for key in sorted(result.keys()):
                    if key.startswith("eval"):
                        tf.logging.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

        tf.logging.info("*****Best eval results: {} from {}  *****".format(best_metric, best_ckpt))

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
            file_based_convert_examples_to_features(predict_examples, label_list,
                                                    FLAGS.max_seq_length_bert, FLAGS.max_seq_length_sbert,
                                                    tokenizer,
                                                    predict_file)
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
            seq_length_bert=FLAGS.max_seq_length_bert,
            seq_length_sbert=FLAGS.max_seq_length_sbert,
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
                                    serving_input_receiver_fn=serving_input_receiver_fn(FLAGS.max_seq_length))
        tf.logging.info("******* Done for exporting pb file***********")


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("kd_weight_logit")
    flags.mark_flag_as_required("use_kd_logit_mse")
    flags.mark_flag_as_required("use_kd_logit_kl")
    flags.mark_flag_as_required("max_seq_length_bert")
    flags.mark_flag_as_required("max_seq_length_sbert")
    flags.mark_flag_as_required("pooling_strategy")
    flags.mark_flag_as_required("init_checkpoint_teacher")
    flags.mark_flag_as_required("init_checkpoint_student")
    flags.mark_flag_as_required("use_kd_att")
    flags.mark_flag_as_required("kd_weight_att")
    flags.mark_flag_as_required("use_all_layer_emb")
    flags.mark_flag_as_required("use_resnet_predict")
    flags.mark_flag_as_required("use_weighted_att")
    tf.app.run()
