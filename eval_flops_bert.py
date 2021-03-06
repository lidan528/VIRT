# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import json
from tensorflow.python.framework import graph_util
import time
import numpy as np


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
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
    "max_seq_length", None,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 1, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
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

flags.DEFINE_string(
    "pooling_strategy", None,
    "cls or mean"
)


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


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
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


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
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
  tokens_b = tokenizer.tokenize(example.text_b)

  # if tokens_b:
  #   # Modifies `tokens_a` and `tokens_b` in place so that the total
  #   # length is less than the specified length.
  #   # Account for [CLS], [SEP], [SEP] with "- 3"
  #   _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  # else:
  #   # Account for [CLS] and [SEP] with "- 2"
  max_token_length = max_seq_length - 3 #[CLS], a,a,a,a, <PAD>,<PAD>, [SEP],,b,b,b,b, <PAD>,<PAD>[SEP]
  max_token_length_a = max_token_length // 2
  max_token_length_b = max_token_length - max_token_length_a
  if len(tokens_a) > max_token_length_a:
    tokens_a = tokens_a[0:max_token_length_a]
  if len(tokens_b) > max_token_length_b:
    tokens_b = tokens_b[0:max_token_length_b]

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
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)   #[CLS], a,a,a
  input_masks = [1] * len(input_ids)    # [CLS], a,a,a
  while len(input_ids) < max_token_length_a + 1:
    input_ids.append(0)
    input_masks.append(0)
    segment_ids.append(0)
  tokens.append("[SEP]")
  input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])     #[CLS], a,a,a,<PAD>,[SEP],
  input_masks.append(1)
  segment_ids.append(0)

  tokens = []
  for token in tokens_b:
    tokens.append(token)
    input_masks.append(1)
    segment_ids.append(1)
  input_ids += tokenizer.convert_tokens_to_ids(tokens)  # [CLS], a,a,a,<PAD>,[SEP], b,b,b
  while len(input_ids) < 1 + max_token_length_a + 1 + max_token_length_b: # [CLS], a,a,a,<PAD>,[SEP], b,b,b,<PAD>,
    input_ids.append(0)
    input_masks.append(0)
    segment_ids.append(1)
  tokens.append("[SEP]")
  input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], a,a,a,<PAD>,[SEP], b,b,b,<PAD>, [SEP]
  input_masks.append(1)
  segment_ids.append(1)


  assert len(input_ids) == max_seq_length
  assert len(input_masks) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in ["[CLS]"]+tokens_a+["[SEP]"]+tokens_b+["[SEP]"]]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_masks]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_masks,
      segment_ids=segment_ids,
      label_id=label_id,
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
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  # output_layer = model.get_pooled_output()
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
      mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
      masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
      sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
      actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
      actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
      output_layer = sum_masked_output_layer / actual_token_nums

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
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)


    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
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

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      flops = tf.profiler.profile(options=tf.profiler.ProfileOptionBuilder.float_operation())
      tf.logging.info(
          'GFLOPs: {}; '.format(flops.total_float_ops / 1000000000.0))
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
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
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
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


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def create_model_metric_mnli(bert_config, input_ids_ph, input_masks_ph, num_labels):
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids_ph,
        use_one_hot_embeddings=FLAGS.use_tpu)
    if FLAGS.pooling_strategy == "cls":
        tf.logging.info("use cls embedding")
        output_layer = model.get_pooled_output()

    elif FLAGS.pooling_strategy == "mean":
        tf.logging.info("use mean embedding")

        output_layer = model.get_sequence_output()

        mask = tf.cast(tf.expand_dims(input_masks_ph, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_masks_ph, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
        output_layer = sum_masked_output_layer / actual_token_nums

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return probabilities


def create_model_metric_squad(bert_config, input_ids_ph, input_masks_ph, num_labels):
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids_ph,
        use_one_hot_embeddings=FLAGS.use_tpu)
    output_layer = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(output_layer, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(output_layer,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
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
        "squad": create_model_metric_squad
    }

    task_name = FLAGS.task_name.lower()
    processor = processors[task_name]() if task_name in processors else None
    metric_func = metric_funcs[task_name]
    label_list = processor.get_labels() if task_name in processors else [0]

    input_ids_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length], dtype=tf.int32, name='input_ids')
    input_masks_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length], dtype=tf.int32, name='input_masks')
    result = metric_func(bert_config, input_ids_ph, input_masks_ph, len(label_list))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        opt_builder = tf.profiler.ProfileOptionBuilder
        prof_options = opt_builder.float_operation()
        prof_options['hide_name_regexes'] = ['.*/Initializer/.*']
        tfprof_node = tf.profiler.profile(sess.graph, options=prof_options)
        tf.logging.info('GFLOPs: {};    '.format(tfprof_node.total_float_ops / 1000000000.0))



def metric_latency(bert_config, batch_num):
    processors = {
        "mnli": MnliProcessor,
        "qqp": QqpProcessor
    }
    metric_funcs = {
        "mnli": create_model_metric_mnli,
        "squad": create_model_metric_squad
    }

    task_name = FLAGS.task_name.lower()
    processor = processors[task_name]() if task_name in processors else None
    metric_func = metric_funcs[task_name]
    label_list = processor.get_labels() if task_name in processors else [0]
    input_ids_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length], dtype=tf.int32,
                                  name='input_ids_ph')
    input_masks_ph = tf.placeholder(shape=[FLAGS.train_batch_size, FLAGS.max_seq_length], dtype=tf.int32,
                                    name='input_masks_ph')
    result = metric_func(bert_config, input_ids_ph, input_masks_ph, len(label_list))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_ids_dataset_np = np.random.randint(0,5000, size=[batch_num, FLAGS.train_batch_size, FLAGS.max_seq_length])
        input_masks_dataset_np = np.random.randint(0,2, size=[batch_num, FLAGS.train_batch_size, FLAGS.max_seq_length])
        t_start = time.time()
        for input_ids_np, input_masks_np in zip(input_ids_dataset_np, input_masks_dataset_np):
            sess.run(result, feed_dict={input_ids_ph : input_ids_np, input_masks_ph : input_masks_np})
        t_end = time.time()

        tf.logging.info('Latency: {};    '.format((t_end-t_start) / batch_num))





def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)


  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  # metric_flops(bert_config)
  metric_latency(bert_config, batch_num=100)



if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  # flags.mark_flag_as_required("pooling_strategy")
  tf.app.run()
