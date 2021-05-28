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
import six
import math
import random
import re
import string

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

# flags.DEFINE_integer(
#     "max_seq_length", None,
#     "The maximum total input sequence length after WordPiece tokenization. "
#     "Sequences longer than this will be truncated, and sequences shorter "
#     "than this will be padded.")

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

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_string(
    "train_file", None,
    "train_jsonl file"
)

flags.DEFINE_string(
    "dev_file", None,
    "dev_jsonl file"
)



class SquadExample(object):
  """A single training/test example for simple sequence classification.
     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample.
  处理好的jsonl文件为若干行，
  每一行为{"id":,  "seq1":(question),  "seq2":(doc),,  "label":{"ans":[[start_idx, content]]}}
  这个数据集一个问题其实只有一个答案
  """
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  with tf.gfile.Open(input_file, "r") as reader:
    for paragraph in reader:
      paragraph = json.loads(paragraph.strip())
      # print(paragraph,'**')
      paragraph_text = paragraph["seq2"]      # 这里一定不能加strip！！！！，如果开头是\n，那么答案的idx就会错位
      doc_tokens = []       # 元素为单词
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)    #前面是空格，代表单词的第一个字母
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)    # 这个字符(包括空格)所在词，在句子中的idx; char_to_(word_offset)，
        ## 因为需要映射到答案idx上去, 所以空格也要保留


      is_impossible = False
      if FLAGS.version_2_with_negative:
        is_impossible = paragraph["label"]["cls"]
      if (len(paragraph["label"]["ans"]) != 1) and (not is_impossible):
        raise ValueError(
          "For training, each question should have exactly 1 answer.")
      # for ans in paragraph["label"]:
      question_text = paragraph["seq1"]
      qas_id = paragraph["id"]
      start_position = None
      end_position = None
      orig_answer_text = None
      # print(char_to_word_offset, '--')
      if is_training:
        if not is_impossible:
          answer = paragraph["label"]["ans"][0]
          orig_answer_text = answer[1]    # 这里也不加strip，因为answer的实际offset可能从空白字符开始, 会影响后面的end_position计算
          answer_offset = answer[0]
          answer_length = len(orig_answer_text)       # 原答案文本的字符数
          start_position = char_to_word_offset[answer_offset]     # 答案的首字符在第几个词
          end_position = char_to_word_offset[answer_offset + answer_length - 1]  #答案的最后一个字符在第几个词
          # Only add answers where the text can be exactly recovered from the
          # document. If this CAN'T happen it's likely due to weird Unicode
          # stuff so we will just skip the example.
          #
          # Note that this means for training mode, every example is NOT
          # guaranteed to be preserved.
          actual_text = " ".join(
              doc_tokens[start_position:(end_position + 1)])          #答案文本中的词列表？（为什么要这样不清楚）
          cleaned_answer_text = " ".join(
              tokenization.whitespace_tokenize(orig_answer_text))     #以空格为分割的答案文本的词列表
          if actual_text.find(cleaned_answer_text) == -1:
            tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                               actual_text, cleaned_answer_text)
            continue
        else:
          start_position = -1
          end_position = -1
          orig_answer_text = ""

      example = SquadExample(
          qas_id=qas_id,
          question_text=question_text,
          doc_tokens=doc_tokens,
          orig_answer_text=orig_answer_text,
          start_position=start_position,  # 第一个字符所在词在原句子中的idx
          end_position=end_position,      # 最后一个字符所在词在原句子中的idx
          is_impossible=is_impossible)
      examples.append(example)

  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]       # 限定了最长的query_length (64)

    tok_to_orig_index = []             # 每个token所在词的idx, 即token_idx --> word_idx
    orig_to_tok_index = []          # 原来的词的第一个token在token_list中的idx
    all_doc_tokens = []             # 原句子中的每个token
    for (i, token) in enumerate(example.doc_tokens):        # doc_tokens:原句子中的每个词
      orig_to_tok_index.append(len(all_doc_tokens))         # 原来的词的第一个token在token_list中的idx
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)                 # 每个token所在词的idx, 即token_idx --> word_idx
        all_doc_tokens.append(sub_token)
    #------这部分全都是建立doc中token的位置映射，无关query------------------------

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]        # 将原答案中第一个词的idx转为第一个token的idx
      if example.end_position < len(example.doc_tokens) - 1:                # 如果答案的最后一个词不是句子的最后一个词
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1  # 最后一个词idx+1：下一个词的idx；将下一个词的idx映射为token_idx, 然后减1即为最后一个
        # 即答案的第一个字符所在的词的词idx(example.start_position), 这个词token后，第一个token的token_idx为tok_start_position
        # 答案的最后一个字符所在的词的下一个词idx(example.end_position+1), 这个词token后，第一个token的(token_idx-1)为tok_end_position, 所以是左闭you闭
      else:     #此时没有下一个词了
        tok_end_position = len(all_doc_tokens) - 1                          # 最后一个token的idx
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    #---------------这部分也是获取答案在整段doc中的token_idx，无关query------------------

    # The -3 accounts for [CLS], [SEP] and [SEP]
    # max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    max_tokens_for_doc = max_seq_length - max_query_length - 3        # fix query length

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))     # 每一段都存着从start_offset起，长为length的tokens段
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)       # 最长不能跳过length (即不能有token没有覆盖到)

    #------------------这部分也是获取span在整段doc中的offset，无关query----------------------

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      input_mask = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      input_mask.append(1)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
        input_mask.append(1)
      # PAD query to max_query_length
      input_ids = tokenizer.convert_tokens_to_ids(tokens)     # [CLS], query_tokens,
      while len(tokens) < 1 + max_query_length:
        tokens.append("[PAD]")
        segment_ids.append(0)
        input_ids.append(0)
        input_mask.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)
      input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], query_tokens, <PAD>,[SEP],
      input_mask.append(1)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i      # 一段中要处理的token的idx为该段的start_offset + 该token在该段中的偏移
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index] #token_to_orig_map{token在query+该段context中的idx : token对应的词在原doc句子中的idx}
            # 也即  {该段doc中的token在这一次整体BERT输入中的idx: token对应的词在原doc句子中的idx}, 注意只对真实存在的span token设置了key
        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context       #{该段doc中的token在这一次整体BERT输入中的idx: 该段span对于这个token来说是不是最居中的}
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
        input_ids += tokenizer.convert_tokens_to_ids([all_doc_tokens[split_token_index]])
        input_mask.append(1)
      #   [CLS], query_tokens, <PAD>,[SEP], doc
      while len(tokens) < max_seq_length - 1:
        tokens.append("[PAD]")
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)
      input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], query_tokens, <PAD>,[SEP], doc, <PAD>, [SEP]
      input_mask.append(1)

      # input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      # input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      # while len(input_ids) < max_seq_length:
      #   input_ids.append(0)
      #   input_mask.append(0)
      #   segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1      # 左闭右闭
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):               # 如果训练时答案没有完全在这一段里面，就舍弃掉这个example
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          # doc_offset = len(query_tokens) + 2
          doc_offset = len(max_query_length) + 2          # fixed query length
          start_position = tok_start_position - doc_start + doc_offset
          # 在这一段BERT输入中的偏移，tok_start_position为该token在整个doc中的token_idx， doc_offset为在doc在BERT输入中的偏移
          # tok_start_position - doc_start 为token在该段span中的偏移, + offset即为在BERT的偏移
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0

      if example_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and example.is_impossible:
          tf.logging.info("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))

      feature = InputFeatures(
          unique_id=unique_id,          # 在块中的unique id
          example_index=example_index,  # 一个answer-question-doc的 id，即来自于同一个doc的span共享example_index
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=start_position,
          end_position=end_position,
          is_impossible=example.is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).   此时虽然答案在最后一个词里，但答案的开始token 1895 并不是最后一个词的开始token （
  #   即常规的start_token是答案的第一个字符所在的词的词idx经过token后，第一个token的token idx
  #   在这个例子中，第一个字符为1，所在词为(1895-1943)，经过token后的第一个token为 (，它的idx显然不能代表1895
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  # 注意进入这个函数只会发生在答案的最后一个字符所在的词为原文的最后一个词，应该是专门适应这种情况；
  # input_start为答案的第一个字符所在的词，产生的第一个token的token_idx
  # input_end为 整句子的最后一个token的idx，也可以理解为最后一个词(即本词)产生的最后一个token的token_idx
  # 因此实际上是在最后一个词中进行寻找token的起止范围
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):           # 此时采用在这个词( 1895 - 1943 )中遍历寻找token
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).  也就是说，让它的位置尽可能居中
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  # doc_spans: 元组列表，(该段span在整个句子中的offset，该段span的长度)
  # cur_span_index: 这是第几段span
  # position: 某个token在整个doc中的位置, 值得一提的是这个token必定在cur_span_index代表的span下，但并不一定只出现在这个span中
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index





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



def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
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

  return (start_logits, end_logits)         # [bs, seq_len]


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
      bert_config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

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
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        "unique_ids": unique_ids,
        "start_logits": start_logits,
        "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
        "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)    # 来自于同一个doc的不同span共享一个example_index

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result          # 将每一个span的独特id映射为对应的答案

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]         # 这个doc下所有span的预测

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):                        # 同一doc的所有span
      result = unique_id_to_result[feature.unique_id]                           # 该span的预测结果
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)       # 该span对于start index的最高的n_best_size
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict           # 同同一doc的所有span，每个span都进行n_best_size X n_best_size 组合，取出其中合法的
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            #{该段doc中的token在这一次整体BERT输入中的idx: token对应的词在原doc句子中的idx}, 如果不在，说明start_index不在doc span中
            #token_to_orig_map 中的key都是真实存在的token，这样过滤掉了padding
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):  # start不是在最居中的位置
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,              # 从0开始, 同一doc中的不同span的idx
                  start_index=start_index,                  # 预测的得分在前n_best_size且合法的，start_token在BERT输入中的index
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],         # 得分值
                  end_logit=result.end_logits[end_index]))
                                                        # 这样最多可以得到 n_best * n_best个区间的预测结果
    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,     # 在0，0这个位置最小的得分的span，在doc中的顺序
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))        # 在该doc的span的预测集合中再加上一条非法的
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),        # 一段doc中所有的合法span预测按照logit预测最大排序
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:                         # 一个doc最多可以得到的n_span * n_best * n_best个区间的预测结果中，只保留n_best_size个span预测
        break
      feature = features[pred.feature_index]                # 选中这一条feature
      if pred.start_index > 0:                              # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]      # 一直是左闭右臂区间, 在该段span中的范围
        orig_doc_start = feature.token_to_orig_map[pred.start_index]            # 也即  {该段doc中的token在这一次整体BERT输入中的idx: token对应的词在原doc句子中的idx}
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]     # 真正的词列表, 但是可能会包含冗余信息, 如: 词是 (1893--1902)
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)                   #答案的所有字符所在在原始句子中的词列表, 可能会包含冗余信息, 如: 词是 (1893--1902)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,                              # 一个doc中最终比较好的span预测的答案文本,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []       # 最终得到的所有span预测中的可能性打分
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)          # 手动对最多 n_best个预测的可能性进行softmax

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not FLAGS.version_2_with_negative:       # 如果无null, 直接返回最大的可能性
      all_predictions[example.qas_id] = nbest_json[0]["text"]       # 在这里记录了原始id
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if FLAGS.version_2_with_negative:
    with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)       #冠词不应该影响答案

  def white_space_fix(text):
    return ' '.join(text.split())                     # 空白符同一换成空格

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)    # 移除标点符号

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


# find longest common string
def find_lcs(s1, s2):
  m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
  mmax = 0
  p = 0
  for i in range(len(s1)):
    for j in range(len(s2)):
      if s1[i] == s2[j]:
        m[i + 1][j + 1] = m[i][j] + 1
        if m[i + 1][j + 1] > mmax:
          mmax = m[i + 1][j + 1]
          p = i + 1
  return s1[p - mmax:p], mmax


# remove punctuation
def remove_punctuation(in_str):
  in_str = str(in_str).lower().strip()
  sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
             '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
             '「', '」', '（', '）', '－', '～', '『', '』']
  out_segs = []
  for char in in_str:
    if char in sp_char:
      continue
    else:
      out_segs.append(char)
  return ''.join(out_segs)


def calc_f1_score(answers, prediction):
  f1_scores = []
  for ans in answers:
    ans_segs = normalize_answer(ans)
    prediction_segs = normalize_answer(prediction)
    lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
    if lcs_len == 0:
      f1_scores.append(0)
      continue
    precision = 1.0 * lcs_len / len(prediction_segs)
    recall = 1.0 * lcs_len / len(ans_segs)
    f1 = (2 * precision * recall) / (precision + recall)
    f1_scores.append(f1)
  return max(f1_scores)


def calc_em_score(answers, prediction):
  em = 0
  for ans in answers:
    ans_ = normalize_answer(ans)
    prediction_ = normalize_answer(prediction)
    if ans_ == prediction_:
      em = 1
      break
  return em


def evaluate(ground_truth_file, prediction_file):
  f1 = 0
  em = 0
  total_count = 0
  skip_count = 0
  for instance in ground_truth_file:
    #处理好的jsonl文件为若干行，
    #每一行为
    #{"id":, "seq1": (question), "seq2": (doc), , "label": {"ans": [[start_idx, content]]}}
    #这个数据集一个问题其实只有一个答案
    total_count += 1
    query_id = instance['id'].strip()
    # query_text = instance['question'].strip()
    answers = [x[1] for x in instance['label']['ans']]

    if query_id not in prediction_file:
      print('Unanswered question: {}\n'.format(query_id))
      skip_count += 1
      continue

    prediction = str(prediction_file[query_id])
    f1 += calc_f1_score(answers, prediction)
    em += calc_em_score(answers, prediction)

  f1_score = 100.0 * f1 / total_count
  em_score = 100.0 * em / total_count
  return f1_score, em_score, total_count, skip_count


def get_eval(original_file, prediction_file):
  original_file_data = []
  with open(original_file, 'r') as ofp:
    for line in ofp:
      line_data = json.loads(line.strip())
      original_file_data.append(line_data)      #因为这次的输入是自定义文件了

  # ground_truth_file = json.load(open(original_file, 'r'))
  prediction_file = json.load(open(prediction_file, 'r'))
  F1, EM, TOTAL, SKIP = evaluate(original_file_data, prediction_file)
  AVG = (EM + F1) * 0.5
  output_result = collections.OrderedDict()
  output_result['AVERAGE'] = '%.3f' % AVG
  output_result['F1'] = '%.3f' % F1
  output_result['EM'] = '%.3f' % EM
  output_result['TOTAL'] = TOTAL
  output_result['SKIP'] = SKIP

  return output_result




def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_eval:
    raise ValueError(
        "At least one of `do_train`, do eval  or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
      keep_checkpoint_max=200,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  if FLAGS.do_train:
    train_examples = read_squad_examples(
      input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file_tfr = os.path.join(FLAGS.output_dir, "train.tf_record")

    train_writer = FeatureWriter(
      filename=train_file_tfr,
      is_training=True)
    convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

    train_input_fn = input_fn_builder(
        input_file=train_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = read_squad_examples(
      input_file=FLAGS.dev_file, is_training=False)

    eval_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
      is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=False,
      output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
      input_file=eval_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.

    steps_and_files = []
    filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
    for filename in filenames:
      if filename.endswith(".index"):
        ckpt_name = filename[:-6]
        cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
        global_step = int(cur_filename.split("-")[-1])
        tf.logging.info("Add {} to eval list.".format(cur_filename))
        steps_and_files.append([global_step, cur_filename])
    steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

    output_eval_file = os.path.join(FLAGS.output_dir, "dev_results.txt")
    print("output_eval_file:", output_eval_file)
    tf.logging.info("output_eval_file:" + output_eval_file)

    with tf.gfile.GFile(output_eval_file, "w") as writer:
      for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
        tf.logging.info("evaluating {}...".format(filename) )
        all_results = []
        for result in estimator.predict(
                predict_input_fn, yield_single_examples=True, checkpoint_path=filename):
          if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
          unique_id = int(result["unique_ids"])
          start_logits = [float(x) for x in result["start_logits"].flat]
          end_logits = [float(x) for x in result["end_logits"].flat]
          all_results.append(
            RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))

        output_prediction_file = os.path.join(FLAGS.output_dir, "dev_predictions.json")
        output_nbest_file = os.path.join(FLAGS.output_dir, "dev_nbest_predictions.json")
        output_null_log_odds_file = os.path.join(FLAGS.output_dir, "dev_null_odds.json")

        write_predictions(eval_examples, eval_features, all_results,
                          FLAGS.n_best_size, FLAGS.max_answer_length,
                          FLAGS.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)

        eval_result = get_eval(FLAGS.dev_file, output_prediction_file)

        tf.logging.info("***** Eval results %s *****" % (filename))
        writer.write("***** Eval results %s *****\n" % (filename))
        for key in sorted(eval_result.keys()):
          tf.logging.info("  %s = %s", key, str(eval_result[key]))
          writer.write("%s = %s\n" % (key, str(eval_result[key])))

  if FLAGS.do_predict:
    eval_examples = read_squad_examples(
      input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
      is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=False,
      output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(
      input_file=eval_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
        RawResult(
          unique_id=unique_id,
          start_logits=start_logits,
          end_logits=end_logits))

    output_prediction_file = os.path.join(FLAGS.output_dir, "test_predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "test_nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "test_null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
