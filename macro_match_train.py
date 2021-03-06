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
    "max_seq_length_query", None,
    "The maximum total input sequence length after WordPiece tokenization. "
    "In the origin bert model"
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_doc", None,
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

flags.DEFINE_integer("poly_first_m", 64, "number of tokens to ue in poly-encoder.")

tf.flags.DEFINE_string(
    "model_type", None,
    "which model to use"
    )

# tf.flags.DEFINE_integer("poly_first_m", 64, "if use poly-encoder, number of document embeddings to choose")

flags.DEFINE_integer("colbert_dim", 128, "reduction dimension of colbert")


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

class BoolqProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_jsnl(os.path.join(data_dir, "boolq-train.jsonl")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_jsnl(os.path.join(data_dir, "boolq-dev.jsonl")),
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


class RTEProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_jsnl(os.path.join(data_dir, "train.jsonl")), "train")

  def get_dev_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_jsnl(os.path.join(data_dir, "val.jsonl")),
          "dev")

  def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_jsnl(os.path.join(data_dir, "test.jsonl")), "test")

  def get_labels(self):
      """See base class."""
      # return ["contradiction", "entailment", "neutral"]
      return ["not_entailment", "entailment"]

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
          text_a = tokenization.convert_to_unicode(line['hypothesis'])
          text_b = tokenization.convert_to_unicode(line['premise'])
          # if set_type == "test":
          #  label = "contradiction"
          # else:
          # label = tokenization.convert_to_unicode(line['label']['cls'])
          label = line['label']
          # if label == tokenization.convert_to_unicode("contradictory"):
          #   label = tokenization.convert_to_unicode("contradiction")
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      return examples


def conver_single_example_distill(ex_index, example, rele_label_list, max_seq_length_query, max_seq_length_doc,
                           tokenizer):
    """
    ??????teacher?????????student??????????????????????????????????????????convert_example???????????????????????????
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
        ?????????????????????????????????att score map??????, ???????????????sen1???sen2????????????;
        ??????????????????, ???sen1???sen2??????????????????????????????max_len
        """
        # ??????: [CLS] a,a,a,a,a,a, [SEP], b, b, b, b ,b [SEP]
        max_seq_length_bert = 1 + max_seq_length_query-2 + 1 + max_seq_length_doc-2 + 1
        max_token_len = max_seq_length_bert - 3
        max_a_len = max_seq_length_query - 2
        max_b_len = max_seq_length_doc - 2
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

    def build_bert_input_s_bert(tokens_temp, max_len):

        if len(tokens_temp) > max_len - 2:
            tokens_temp = tokens_temp[0: (max_len - 2)]

        tokens_p = []
        segment_ids = []

        tokens_p.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_temp:
            tokens_p.append(token)
            segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_p)  # [CLS], a,a,a
        input_mask = [1] * len(input_ids)  # [CLS], a,a,a

        while len(input_ids) < max_len-1: # [CLS], a,a,a,<PAD>,
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        tokens_p.append("[SEP]")
        input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])     # [CLS], a,a,a,<PAD>, [SEP]
        input_mask.append(1)
        segment_ids.append(0)

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len
        return input_ids, input_mask, segment_ids

    input_ids_bert_ab, input_mask_bert_ab, segment_ids_bert_ab = build_bert_input_bert(tokens_a, tokens_b)
    input_ids_sbert_a, input_mask_sbert_a, segment_ids_sbert_a = build_bert_input_s_bert(tokens_a, max_seq_length_query)
    input_ids_sbert_b, input_mask_sbert_b, segment_ids_sbert_b = build_bert_input_s_bert(tokens_b, max_seq_length_doc)

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
        examples, label_list, max_seq_length_query, max_seq_length_doc, tokenizer, output_file, is_training=False):
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
                                         max_seq_length_query, max_seq_length_doc, tokenizer)

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


def file_based_input_fn_builder(input_file, seq_length_query, seq_length_doc,
                                is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    seq_length_bert = 1 + seq_length_query-2 + 1 + seq_length_doc-2 + 1
    name_to_features = {
        "input_ids_sbert_a": tf.FixedLenFeature([seq_length_query], tf.int64),
        "input_mask_sbert_a": tf.FixedLenFeature([seq_length_query], tf.int64),
        "segment_ids_sbert_a": tf.FixedLenFeature([seq_length_query], tf.int64),

        "input_ids_sbert_b": tf.FixedLenFeature([seq_length_doc], tf.int64),
        "input_mask_sbert_b": tf.FixedLenFeature([seq_length_doc], tf.int64),
        "segment_ids_sbert_b": tf.FixedLenFeature([seq_length_doc], tf.int64),

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
                       segment_ids, use_one_hot_embeddings, scope, is_reuse, pooling=False):
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
    if not pooling:
        output_layer = model.get_sequence_output()
    else:
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


def create_model_Deformer(bi_layer_num, bert_config, is_training,
                          input_ids_a, input_mask_a, segment_ids_a,
                          input_ids_b, input_mask_b, segment_ids_b,
                          use_one_hot_embeddings):
    import copy
    bert_config_a = copy.deepcopy(bert_config)
    total_layer_num = bert_config.num_hidden_layers
    bert_config_a.num_hidden_layers = bi_layer_num
    query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config_a, is_training=is_training,
                                                          input_ids=input_ids_a,
                                                          input_mask=input_mask_a,
                                                          segment_ids=segment_ids_a,
                                                          use_one_hot_embeddings=use_one_hot_embeddings,
                                                          scope="bert_student",
                                                          is_reuse=tf.AUTO_REUSE,
                                                          pooling=False)
    doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config_a, is_training=is_training,
                                                      input_ids=input_ids_b,
                                                      input_mask=input_mask_b,
                                                      segment_ids=segment_ids_b,
                                                      use_one_hot_embeddings=use_one_hot_embeddings,
                                                      scope="bert_student",
                                                      is_reuse=tf.AUTO_REUSE,
                                                      pooling=False)
    combined_embeddings = tf.concat([query_embedding, doc_embedding[, 1:, :]], axis=1)     #[bs, seq_len ,emb_dim], remove cls in doc
    combined_input_masks = tf.concat([input_mask_a, input_mask_b[, 1:]], axis=1)        #[bs, seq_len]
    combined_att_masks = create_att_mask_for1(combined_input_masks)                    #[bs, seq_len, seq_len]
    input_shape = modeling.get_shape_list(combined_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]
    combined_embeddings = modeling.reshape_to_matrix(combined_embeddings)
    # print("seq_len:>>>>>>>>>>", seq_len)
    # input_shape = modeling.get_shape_list(combined_embeddings, expected_rank=[3])
    prev_output = combined_embeddings
    all_upper_layer_output = []
    with tf.variable_scope("bert_student", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("encoder"):
            for layer_idx in range(bi_layer_num, total_layer_num):
                with tf.variable_scope("layer_%d" % bi_layer_num):
                    layer_input = prev_output
                    with tf.variable_scope("attention"):
                        with tf.variable_scope("self"):
                            self_att_output, _, _, _, _ = modeling.attention_layer(
                                from_tensor=layer_input,
                                to_tensor=layer_input,
                                attention_mask=combined_att_masks,
                                num_attention_heads=bert_config.num_attention_heads,
                                size_per_head=int(bert_config.hidden_size / bert_config.num_attention_heads),
                                attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                                initializer_range=bert_config.initializer_range,
                                do_return_2d_tensor=True,
                                batch_size=batch_size,
                                from_seq_length=seq_length,
                                to_seq_length=seq_length
                            )
                        with tf.variable_scope("output"):
                            attention_output = tf.layers.dense(
                                self_att_output,
                                bert_config.hidden_size,
                                kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
                            attention_output = modeling.dropout(attention_output, bert_config.hidden_dropout_prob)
                            attention_output = modeling.layer_norm(
                                attention_output + layer_input)  # [bs * seq_len, num_heads * head_dim] (2d)

                        # The activation is only applied to the "intermediate" hidden layer.
                    with tf.variable_scope("intermediate"):
                        intermediate_output = tf.layers.dense(
                            attention_output,
                            bert_config.intermediate_size,
                            activation=modeling.get_activation(bert_config.hidden_act),
                            kernel_initializer=modeling.create_initializer(
                                bert_config.initializer_range))  # [bs * seq_len, num_heads * head_dim] (2d)

                    # Down-project back to `hidden_size` then add the residual.
                    with tf.variable_scope("output"):
                        layer_output = tf.layers.dense(
                            intermediate_output,
                            bert_config.hidden_size,
                            kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
                        layer_output = modeling.dropout(layer_output, bert_config.hidden_dropout_prob)
                        layer_output = modeling.layer_norm(layer_output + attention_output)
                        all_upper_layer_output.append(modeling.reshape_from_matrix(layer_output, input_shape))
                        prev_output = layer_output
            final_output = modeling.reshape_from_matrix(layer_output, input_shape)

    mean_pooled_final_output = get_pooled_embeddings(final_output, combined_input_masks)

    return mean_pooled_final_output, model_stu_query, model_stu_doc, all_upper_layer_output


def create_model_dipair(bi_layer_num, cross_layer_num, bert_config, is_training,
                          input_ids_a, input_mask_a, segment_ids_a,
                          input_ids_b, input_mask_b, segment_ids_b,
                          use_one_hot_embeddings, first_m, first_n):
    import copy
    bert_config_a = copy.deepcopy(bert_config)
    bert_config_a.num_hidden_layers = bi_layer_num
    query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config_a, is_training=is_training,
                                                          input_ids=input_ids_a,
                                                          input_mask=input_mask_a,
                                                          segment_ids=segment_ids_a,
                                                          use_one_hot_embeddings=use_one_hot_embeddings,
                                                          scope="bert_student",
                                                          is_reuse=tf.AUTO_REUSE,
                                                          pooling=False)
    doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config_a, is_training=is_training,
                                                      input_ids=input_ids_b,
                                                      input_mask=input_mask_b,
                                                      segment_ids=segment_ids_b,
                                                      use_one_hot_embeddings=use_one_hot_embeddings,
                                                      scope="bert_student",
                                                      is_reuse=tf.AUTO_REUSE,
                                                      pooling=False)
    query_embedding = query_embedding[:, :first_m, :]
    doc_embedding = doc_embedding[:, :first_n, :]
    input_mask_a = input_mask_a[:, :first_m]
    input_mask_b = input_mask_b[:, :first_n]
    combined_embeddings = tf.concat([query_embedding, doc_embedding], axis=1)  # [bs, seq_len ,emb_dim]
    combined_input_masks = tf.concat([input_mask_a, input_mask_b], axis=1)  # [bs, seq_len]
    combined_att_masks = create_att_mask_for1(combined_input_masks)  # [bs, seq_len, seq_len]
    bs, seq_len, _ = modeling.get_shape_list(combined_embeddings, expected_rank=[3])
    input_shape = modeling.get_shape_list(combined_embeddings, expected_rank=3)
    combined_embeddings = modeling.reshape_to_matrix(combined_embeddings)

    with tf.variable_scope("bert_student", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("encoder"):
            with tf.variable_scope("layer_%d" % bi_layer_num):
                with tf.variable_scope("attention"):
                    with tf.variable_scope("self"):
                        self_att_output, _, _, _, _ = modeling.attention_layer(
                            from_tensor=combined_embeddings,
                            to_tensor=combined_embeddings,
                            attention_mask=combined_att_masks,
                            num_attention_heads=bert_config.num_attention_heads,
                            size_per_head=int(bert_config.hidden_size / bert_config.num_attention_heads),
                            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                            initializer_range=bert_config.initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=bs,
                            from_seq_length=seq_len,
                            to_seq_length=seq_len
                        )
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            self_att_output,
                            bert_config.hidden_size,
                            kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
                        attention_output = modeling.dropout(attention_output, bert_config.hidden_dropout_prob)
                        attention_output = modeling.layer_norm(
                            attention_output + combined_embeddings)  # [bs * seq_len, num_heads * head_dim] (2d)

                    # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        bert_config.intermediate_size,
                        activation=modeling.get_activation(bert_config.hidden_act),
                        kernel_initializer=modeling.create_initializer(
                            bert_config.initializer_range))  # [bs * seq_len, num_heads * head_dim] (2d)

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense(
                        intermediate_output,
                        bert_config.hidden_size,
                        kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
                    layer_output = modeling.dropout(layer_output, bert_config.hidden_dropout_prob)
                    layer_output = modeling.layer_norm(layer_output + attention_output)
                    final_output = modeling.reshape_from_matrix(layer_output, input_shape)

    mean_pooled_final_output = get_pooled_embeddings(final_output, combined_input_masks)

    return mean_pooled_final_output, model_stu_query, model_stu_doc




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

        if FLAGS.model_type == 'poly':
            tf.logging.info("*********** use poly encoder as the model backbone...*******************")
            query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                                  input_ids=input_ids_sbert_a,
                                                                  input_mask=input_mask_sbert_a,
                                                                  segment_ids=segment_ids_sbert_a,
                                                                  use_one_hot_embeddings=use_one_hot_embeddings,
                                                                  scope="bert_student",
                                                                  is_reuse=tf.AUTO_REUSE,
                                                                  pooling=True)
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=False)
            regular_embedding = poly_encoder(query_embedding, doc_embedding, input_mask_sbert_b, bert_config)

        elif FLAGS.model_type == 'deformer':
            tf.logging.info("*********** use deformer as the model backbone...*******************")
            regular_embedding, model_stu_query, model_stu_doc, all_upper_layer_output = create_model_Deformer(bi_layer_num=11,
                                                                                      bert_config=bert_config, is_training=is_training,
                                                                                      input_ids_a=input_ids_sbert_a,
                                                                                      input_mask_a=input_mask_sbert_a,
                                                                                      segment_ids_a=segment_ids_sbert_a,
                                                                                      input_ids_b=input_ids_sbert_b,
                                                                                      input_mask_b=input_mask_sbert_b,
                                                                                      segment_ids_b=segment_ids_sbert_b,
                                                                                      use_one_hot_embeddings=use_one_hot_embeddings)

        elif FLAGS.model_type == 'dipair':
            tf.logging.info("*********** use dipair as the model backbone...*******************")
            regular_embedding, model_stu_query, model_stu_doc = create_model_dipair(bi_layer_num=11,
                                                                                      cross_layer_num=1,
                                                                                      bert_config=bert_config,
                                                                                      is_training=is_training,
                                                                                      input_ids_a=input_ids_sbert_a,
                                                                                      input_mask_a=input_mask_sbert_a,
                                                                                      segment_ids_a=segment_ids_sbert_a,
                                                                                      input_ids_b=input_ids_sbert_b,
                                                                                      input_mask_b=input_mask_sbert_b,
                                                                                      segment_ids_b=segment_ids_sbert_b,
                                                                                      use_one_hot_embeddings=use_one_hot_embeddings,
                                                                                      first_m=8,
                                                                                      first_n=16)

        elif FLAGS.model_type == 'col':
            tf.logging.info("*********** use colbert as the model backbone...*******************")
            query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                                  input_ids=input_ids_sbert_a,
                                                                  input_mask=input_mask_sbert_a,
                                                                  segment_ids=segment_ids_sbert_a,
                                                                  use_one_hot_embeddings=use_one_hot_embeddings,
                                                                  scope="bert_student",
                                                                  is_reuse=tf.AUTO_REUSE,
                                                                  pooling=False)
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=False)
            logits_student, probabilities_student, log_probs_student = col_bert(query_embedding=query_embedding,
                                                                                doc_embedding=doc_embedding,
                                                                                input_mask_a=input_mask_sbert_a,
                                                                                input_mask_b=input_mask_sbert_b,
                                                                                num_rele_label=num_rele_label,
                                                                                bert_config=bert_config)

        elif FLAGS.model_type == 'late_fusion':
            tf.logging.info("*********** use late fusion as the model backbone...*******************")
            query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                                  input_ids=input_ids_sbert_a,
                                                                  input_mask=input_mask_sbert_a,
                                                                  segment_ids=segment_ids_sbert_a,
                                                                  use_one_hot_embeddings=use_one_hot_embeddings,
                                                                  scope="bert_student",
                                                                  is_reuse=tf.AUTO_REUSE,
                                                                  pooling=False)
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=False)
            regular_embedding = newly_late_interaction(query_embedding, input_mask_sbert_a, doc_embedding, input_mask_sbert_b)

        else:
            query_embedding, model_stu_query = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                                  input_ids=input_ids_sbert_a,
                                                                  input_mask=input_mask_sbert_a,
                                                                  segment_ids=segment_ids_sbert_a,
                                                                  use_one_hot_embeddings=use_one_hot_embeddings,
                                                                  scope="bert_student",
                                                                  is_reuse=tf.AUTO_REUSE,
                                                                  pooling=True)
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=True)
            if FLAGS.model_type == 'sbert':
                tf.logging.info("*********** use sbert as the model backbone...*******************")
                sub_embedding = tf.abs(query_embedding - doc_embedding)
                max_embedding = tf.square(tf.reduce_max([query_embedding, doc_embedding], axis=0))
                regular_embedding = tf.concat([query_embedding, doc_embedding, sub_embedding, max_embedding], -1)
            elif FLAGS.model_type == 'bi_encoder':
                tf.logging.info("*********** use bi-encoder as the model backbone...*******************")
                regular_embedding = tf.concat([query_embedding, doc_embedding], -1)

        if FLAGS.model_type != 'col':
            if FLAGS.model_type == 'late_fusion':
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
            tf.summary.scalar("logit_loss_mse_scaled", scaled_logit_loss)
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
            distill_loss_att = get_attention_loss(model_student_query=model_stu_query,
                                                  model_student_doc=model_stu_doc,
                                                  model_teacher=model_teacher,
                                                  input_mask_sbert_query=input_mask_sbert_a,
                                                  input_mask_sbert_doc=input_mask_sbert_b)
            scaled_att_loss = FLAGS.kd_weight_att * distill_loss_att
            total_loss = total_loss + scaled_att_loss
            tf.summary.scalar("att_loss", distill_loss_att)
            tf.summary.scalar("att_loss_scaled", scaled_att_loss)


        if FLAGS.model_type == 'deformer':
            # deformer???????????????hidden??????
            tf.logging.info('*****use upper hidden layer as distill object...')
            distill_hidden_loss = get_hidden_distill_loss_4_deformer(teacher_model=model_teacher,
                                                                     all_upper_layer_output_deformer=all_upper_layer_output)
            scaled_hidden_loss = distill_hidden_loss * 0.4
            total_loss = total_loss + scaled_hidden_loss
            tf.summary.scalar("hidden_loss", distill_hidden_loss)
            tf.summary.scalar("hidden_loss_scaled", scaled_hidden_loss)


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
    ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    ??????????????????<classifier_bert_bipartition>????????????,
    -------------------------------------------------------------------------------
                 | ckpt??????????????????       |       ????????????????????????
    --------------------------------------------------------------------------------
    BERT?????????    |   bert/....          |      bert_teacher/....
    --------------------------------------------------------------------------------
    ????????????????????? | output_weights,_bias |      cls_teacher/output_weights, _bias
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
    ???????????????????????????, ???????????????????????????????????????????????????????????????BERT?????????????????????
    -------------------------------------------------------------------------------------------
                 | bert_base???ckpt??????????????????       |       ????????????????????????
    -------------------------------------------------------------------------------------------
    BERT?????????    |   bert/....                     |      bert_student/....
    --------------------------------------------------------------------------------------------
    ????????????????????? |      ???????????????????????????           |      cls_student/output_weights, _bias
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


def poly_encoder(query_embedding, doc_embedding, input_mask_b, bert_config):
    import math
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

    doc_embedding = doc_embedding[:, :FLAGS.poly_first_m, :]
    query_embedding = tf.expand_dims(query_embedding, axis=[1])
    poly_mask = input_mask_b[:, :FLAGS.poly_first_m]
    final_vecs = dot_attention(query_embedding, doc_embedding, doc_embedding, v_mask=poly_mask)
    final_vecs = tf.squeeze(final_vecs, axis=[1])  # query????????????????????????mean pooling???

    return final_vecs


def col_bert(query_embedding, doc_embedding, input_mask_a, input_mask_b,  num_rele_label, bert_config):
    import math
    def max_attention_score(q, k):
        # q [B, S, num_label, H], v [B, T, num_label, H]
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        print(modeling.get_shape_list(q))
        attention_scores = tf.matmul(q, k, transpose_b=True)
        # attention_scores [B, num_label, S, T]
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(bert_config.hidden_size)))
        attention_scores = tf.reduce_sum(tf.reduce_max(attention_scores, axis=-1), axis=-1)
        print(modeling.get_shape_list(attention_scores))
        return attention_scores

    query_embedding = tf.layers.dense(query_embedding, units=FLAGS.colbert_dim)
    doc_embedding = tf.layers.dense(doc_embedding, units=FLAGS.colbert_dim)
    B, S, H = modeling.get_shape_list(query_embedding)
    _, T, H = modeling.get_shape_list(doc_embedding)

    transform_weights = tf.get_variable(
        "output_weights", [num_rele_label * FLAGS.colbert_dim, FLAGS.colbert_dim],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    query_embedding = tf.reshape(tf.matmul(query_embedding, transform_weights, transpose_b=True),
                                 [B, S, num_rele_label, H])
    doc_embedding = tf.reshape(tf.matmul(doc_embedding, transform_weights, transpose_b=True), [B, T, num_rele_label, H])

    query_embedding, _ = tf.linalg.normalize(query_embedding, ord=2, axis=-1)
    doc_embedding, _ = tf.linalg.normalize(doc_embedding, ord=2, axis=-1)

    query_mask = tf.expand_dims(input_mask_a, axis=-1)
    query_mask = tf.expand_dims(query_mask, axis=-1)
    query_mask = tf.tile(query_mask, tf.constant([1, 1, num_rele_label, FLAGS.colbert_dim]))
    query_mask = tf.cast(query_mask, dtype=tf.float32)
    query_embedding = tf.multiply(query_mask, query_embedding)

    doc_mask = tf.expand_dims(input_mask_b, axis=-1)
    doc_mask = tf.expand_dims(doc_mask, axis=-1)
    doc_mask = tf.tile(doc_mask, tf.constant([1, 1, num_rele_label, FLAGS.colbert_dim]))
    doc_embedding = tf.multiply(tf.cast(doc_mask, dtype=tf.float32), doc_embedding)

    logits = max_attention_score(query_embedding, doc_embedding)

    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    return logits, probabilities, log_probs


def create_att_mask_for2(input_mask_1, input_mask_2):
  """
  ????????????input_mask????????????seq_len???
  ??????????????????, ?????????input_mask??????i????????????0?????????????????????i????????????0
  ????????????input_mask???0????????????att???????????????att to????????????
  ??????????????????mask_doc?????????????????????query_length????????????[query_length , mask_doc]???mask
  """
  from_shape = modeling.get_shape_list(input_mask_1, expected_rank=2)  # [batch-size, seq_len]
  batch_size = from_shape[0]
  seq_length_self = from_shape[1]
  to_shape = modeling.get_shape_list(input_mask_2, expected_rank=2)
  seq_length_another = to_shape[1]
  mask = tf.cast(
    tf.reshape(input_mask_2, [batch_size, 1, seq_length_another]), tf.float32)
  broadcast_ones = tf.ones(
    shape=[batch_size, seq_length_self, 1], dtype=tf.float32)

  mask = broadcast_ones * mask        #[batch_size, seq_length_another, seq_length_self]
  return mask


def newly_late_interaction(query_embeddings, query_mask, doc_embeddings, doc_mask):
    # query_embeddings: [bs, query_len, emb]
    # doc_embeddings: [bs, doc_len, emb]
    emb_dim = modeling.get_shape_list(query_embeddings, expected_rank=3)[-1]
    query2doc_att = tf.matmul(query_embeddings, doc_embeddings, transpose_b=True)   # [bs, query_len, doc_len]
    query2doc_att = tf.multiply(query2doc_att,
                                   1.0 / math.sqrt(float(emb_dim)))
    query2doc_mask = create_att_mask_for2(query_mask, doc_mask)         # [bs, query_len, doc_len]
    adder1 = (1.0 - tf.cast(query2doc_mask, tf.float32)) * -10000.0
    query2doc_scores = query2doc_att + adder1
    query2doc_probs = tf.nn.softmax(query2doc_scores)       # [bs, query_len, doc_len]
    weighted_doc_embedding = tf.matmul(query2doc_probs, doc_embeddings)     # [bs,  query_len, emb]
    embedding1 = tf.reduce_mean(weighted_doc_embedding, axis=1)

    doc2query_att = tf.matmul(doc_embeddings, query_embeddings, transpose_b=True)
    doc2query_att = tf.multiply(doc2query_att,
                                1.0 / math.sqrt(float(emb_dim)))
    doc2query_mask = create_att_mask_for2(doc_mask, query_mask)
    adder2 = (1.0 - tf.cast(doc2query_mask, tf.float32)) * -10000.0
    doc2query_scores = doc2query_att + adder2
    doc2query_probs = tf.nn.softmax(doc2query_scores)       # [bs, doc_len, query_len]
    weighted_doc_embedding = tf.matmul(doc2query_probs, query_embeddings)   # [bs,  doc_len, emb]
    embedding2 = tf.reduce_mean(weighted_doc_embedding, axis=1)

    sub_embedding = tf.abs(embedding1 - embedding2)
    max_embedding = tf.square(tf.reduce_max([embedding1, embedding2], axis=0))
    regular_embedding = tf.concat([embedding1, embedding2, sub_embedding, max_embedding], -1)

    return regular_embedding


def create_att_mask(input_mask, seq_length_another):
  """
  ????????????input_mask????????????seq_len???
  ??????????????????, ?????????input_mask??????i????????????0?????????????????????i????????????0
  ????????????input_mask???0????????????att???????????????att to????????????
  ??????????????????mask_doc?????????????????????query_length????????????[query_length , mask_doc]???mask
  """
  to_shape = modeling.get_shape_list(input_mask, expected_rank=2)  # [batch-size, seq_len]
  batch_size = to_shape[0]
  seq_length_self = to_shape[1]
  mask = tf.cast(
    tf.reshape(input_mask, [batch_size, 1, seq_length_self]), tf.float32)
  broadcast_ones = tf.ones(
    shape=[batch_size, seq_length_another, 1], dtype=tf.float32)

  mask = broadcast_ones * mask        #[batch_size, seq_length_another, seq_length_self]
  return mask


def create_att_mask_for1(input_mask):
    """
    ????????????input_mask????????????seq_len???
    ??????????????????, ?????????input_mask??????i????????????0?????????????????????i????????????0
    ????????????input_mask???0????????????att???????????????att to????????????
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
    ???????????????attention loss
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
        query_doc_att_matrix_mask = create_att_mask(input_mask_sbert_doc, FLAGS.max_seq_length_query)    # doc??????padding??????????????????attend, [bs, 130, 130]
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
        doc_query_att_matrix_mask = create_att_mask(input_mask_sbert_query, FLAGS.max_seq_length_doc)
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


def get_hidden_distill_loss_4_deformer(teacher_model, all_upper_layer_output_deformer):
    upper_layer_num = len(all_upper_layer_output_deformer)
    all_teacher_layers = teacher_model.all_encoder_layers
    all_upper_layer_output_teacher = all_teacher_layers[-upper_layer_num:]
    loss= 0
    for teacher_layer, deformer_layer in zip(all_upper_layer_output_teacher, all_upper_layer_output_deformer):
        # each layer is [bs, seq_len, emb_dim]
        loss += tf.reduce_sum(tf.square(tf.abs(teacher_layer-deformer_layer)))
    return loss


def get_pooled_embeddings(encode_layer, input_mask):
    """
    ??????mean pool?????????????????????input???padding???(mask???0)????????????
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
        "qqp": QqpProcessor,
        "boolq": BoolqProcessor,
        "rte": RTEProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_save:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # if FLAGS.max_seq_length_bert > bert_config.max_position_embeddings:
    #     raise ValueError(
    #         "Cannot use sequence length %d because the BERT model "
    #         "was only trained up to sequence length %d" %
    #         (FLAGS.max_seq_length, bert_config.max_position_embeddings))

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
                train_examples, label_list, FLAGS.max_seq_length_query, FLAGS.max_seq_length_doc,
                tokenizer, train_file, is_training=True)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length_query=FLAGS.max_seq_length_query,
            seq_length_doc=FLAGS.max_seq_length_doc,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length_query, FLAGS.max_seq_length_doc,
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
            seq_length_query=FLAGS.max_seq_length_query,
            seq_length_doc=FLAGS.max_seq_length_doc,
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
                                                    FLAGS.max_seq_length_query, FLAGS.max_seq_length_doc,
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
            seq_length_query=FLAGS.max_seq_length_query,
            seq_length_doc=FLAGS.max_seq_length_doc,
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
    # flags.mark_flag_as_required("max_seq_length_bert")
    # flags.mark_flag_as_required("max_seq_length_sbert")
    flags.mark_flag_as_required("pooling_strategy")
    flags.mark_flag_as_required("init_checkpoint_teacher")
    flags.mark_flag_as_required("init_checkpoint_student")
    flags.mark_flag_as_required("use_kd_att")
    flags.mark_flag_as_required("kd_weight_att")
    tf.app.run()
