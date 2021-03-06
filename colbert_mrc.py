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
    "max_doc_length", 317,  # 384-64-3 in bert
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "colbert_dim", 128,
    "")


flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_integer(
    "poly_m", 16, "keep poly_m embeddings of the document.")

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
                 tokens_query,
                 tokens_doc,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids_query,
                 input_ids_doc,
                 input_mask_query,
                 input_mask_doc,
                 segment_ids_query,
                 segment_ids_doc,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens_query = tokens_query
        self.tokens_doc = tokens_doc
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids_query = input_ids_query
        self.input_ids_doc = input_ids_doc
        self.input_mask_query = input_mask_query
        self.input_mask_doc = input_mask_doc
        self.segment_ids_query = segment_ids_query
        self.segment_ids_doc = segment_ids_doc
        self.start_position = start_position  # ?????????position?????????????????????????????????token?????????
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample.
  ????????????jsonl?????????????????????
  ????????????{"id":,  "seq1":(question),  "seq2":(doc),,  "label":{"ans":[[start_idx, content]]}}
  ???????????????????????????????????????????????????
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
            paragraph_text = paragraph["seq2"]  # ?????????????????????strip??????????????????????????????\n??????????????????idx????????????
            doc_tokens = []  # ???????????????
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)  # ????????????????????????????????????????????????
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)  # ????????????(????????????)???????????????????????????idx; char_to_(word_offset)???
                ## ???????????????????????????idx??????, ????????????????????????

            is_impossible = paragraph["label"]["cls"]
            # if FLAGS.version_2_with_negative:
            #   is_impossible = paragraph["label"]["cls"]
            if (len(paragraph["label"]["ans"]) == 0):
                raise ValueError(
                    "For training, each question should have exactly 1 answer.")
            # for ans in paragraph["label"]:
            question_text = paragraph["seq1"]
            qas_id = paragraph["id"]
            start_position = None
            end_position = None
            orig_answer_text = None
            # print(char_to_word_offset, '--')
            answers_all = paragraph["label"]["ans"]
            for answer in answers_all:
                if is_training:
                    orig_answer_text = answer[1]  # ???????????????strip?????????answer?????????offset???????????????????????????, ??????????????????end_position??????
                    answer_offset = answer[0]
                    answer_length = len(orig_answer_text)  # ???????????????????????????
                    start_position = char_to_word_offset[answer_offset]  # ?????????????????????????????????
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]  # ??????????????????????????????????????????
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position:(end_position + 1)])  # ???????????????????????????????????????????????????????????????
                    cleaned_answer_text = " ".join(
                        tokenization.whitespace_tokenize(orig_answer_text))  # ?????????????????????????????????????????????
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
                    start_position=start_position,  # ??????????????????????????????????????????idx
                    end_position=end_position,  # ?????????????????????????????????????????????idx
                    is_impossible=is_impossible)
                examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        orig_answer_text = example.orig_answer_text
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]  # ??????????????????query_length (64)

        tok_to_orig_index = []  # ??????token????????????idx, ???token_idx --> word_idx
        orig_to_tok_index = []  # ????????????????????????token???token_list??????idx
        all_doc_tokens = []  # ?????????????????????token
        for (i, token) in enumerate(example.doc_tokens):  # doc_tokens:????????????????????????
            orig_to_tok_index.append(len(all_doc_tokens))  # ????????????????????????token???token_list??????idx
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)  # ??????token????????????idx, ???token_idx --> word_idx
                all_doc_tokens.append(sub_token)
        # ------????????????????????????doc???token????????????????????????query------------------------

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]  # ??????????????????????????????idx???????????????token???idx
            if example.end_position < len(example.doc_tokens) - 1:  # ????????????????????????????????????????????????????????????
                tok_end_position = orig_to_tok_index[
                                       example.end_position + 1] - 1  # ???????????????idx+1??????????????????idx?????????????????????idx?????????token_idx, ?????????1??????????????????
                # ?????????????????????????????????????????????idx(example.start_position), ?????????token???????????????token???token_idx???tok_start_position
                # ??????????????????????????????????????????????????????idx(example.end_position+1), ?????????token???????????????token???(token_idx-1)???tok_end_position, ???????????????you???
            else:  # ???????????????????????????
                tok_end_position = len(all_doc_tokens) - 1  # ????????????token???idx
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # ---------------????????????????????????????????????doc??????token_idx?????????query------------------

        # The -3 accounts for [CLS], [SEP] and [SEP]
        # max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        max_tokens_for_doc = max_seq_length - max_query_length - 3  # fix query length
        assert max_tokens_for_doc == FLAGS.max_doc_length

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
            doc_spans.append(_DocSpan(start=start_offset, length=length))  # ?????????????????????start_offset????????????length???tokens???
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)  # ??????????????????length (????????????token???????????????)

        # ------------------?????????????????????span?????????doc??????offset?????????query----------------------

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens_query = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids_query = []
            input_mask_query = []
            tokens_query.append("[CLS]")
            segment_ids_query.append(0)
            input_mask_query.append(1)
            for token in query_tokens:
                tokens_query.append(token)
                segment_ids_query.append(0)
                input_mask_query.append(1)
            # PAD query to max_query_length
            input_ids_query = tokenizer.convert_tokens_to_ids(tokens_query)  # [CLS], query_tokens,
            while len(tokens_query) < 1 + max_query_length:
                tokens_query.append("[PAD]")
                segment_ids_query.append(0)
                input_ids_query.append(0)
                input_mask_query.append(0)
            tokens_query.append("[SEP]")
            segment_ids_query.append(0)
            input_ids_query += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], query_tokens, <PAD>,[SEP],
            input_mask_query.append(1)
            # -------------------------------------------------------------------------
            tokens_doc = []
            segment_ids_doc = []
            input_mask_doc = []
            input_ids_doc = []
            tokens_doc.append("[CLS]")
            segment_ids_doc.append(1)
            input_mask_doc.append(1)
            input_ids_doc += tokenizer.convert_tokens_to_ids(["[CLS]"])
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i  # ?????????????????????token???idx????????????start_offset + ???token?????????????????????
                token_to_orig_map[len(tokens_doc)] = tok_to_orig_index[
                    split_token_index]  # token_to_orig_map{token???query+??????context??????idx : token??????????????????doc????????????idx}
                # ??????  {??????doc??????token??????????????????BERT????????????idx: token??????????????????doc????????????idx}, ???????????????????????????span token?????????key
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[
                    len(tokens_doc)] = is_max_context  # {??????doc??????token??????????????????BERT????????????idx: ??????span????????????token???????????????????????????}
                tokens_doc.append(all_doc_tokens[split_token_index])
                segment_ids_doc.append(1)
                input_ids_doc += tokenizer.convert_tokens_to_ids([all_doc_tokens[split_token_index]])
                input_mask_doc.append(1)
            #   [CLS], doc_tokens, <PAD>,
            while len(tokens_doc) < max_tokens_for_doc + 1:
                tokens_doc.append("[PAD]")
                input_ids_doc.append(0)
                input_mask_doc.append(0)
                segment_ids_doc.append(1)
            tokens_doc.append("[SEP]")
            segment_ids_doc.append(1)
            input_ids_doc += tokenizer.convert_tokens_to_ids(
                ["[SEP]"])  # [CLS], query_tokens, <PAD>,[SEP], doc, <PAD>, [SEP]
            input_mask_doc.append(1)

            # input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            # input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            # while len(input_ids) < max_seq_length:
            #   input_ids.append(0)
            #   input_mask.append(0)
            #   segment_ids.append(0)

            assert len(input_ids_query) == max_query_length + 2  # [CLS], query, <PAD>, [SEP]
            assert len(input_mask_query) == max_query_length + 2
            assert len(segment_ids_query) == max_query_length + 2

            assert len(input_ids_doc) == FLAGS.max_doc_length + 2  # [CLS], doc, <PAD>, [SEP]
            assert len(input_mask_doc) == FLAGS.max_doc_length + 2
            assert len(segment_ids_doc) == FLAGS.max_doc_length + 2

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1  # ????????????
                out_of_span = False
                if not (
                        tok_start_position >= doc_start and  # tok_start_position: ?????????????????????????????????????????????token???whole doc tokens??????idx
                        tok_end_position <= doc_end):  # ????????????????????????????????????????????????????????????????????????example
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    # doc_offset = len(query_tokens) + 2         # ?????????len(query_tokens) + 2(cls,sep)?????????
                    # doc_offset = max_query_length + 2          # fixed query length
                    doc_offset = 1  # ?????????????????? [CLS]
                    start_position = tok_start_position - doc_start + doc_offset
                    # tok_start_position: ?????????????????????????????????????????????token???whole doc tokens??????idx
                    # doc_start;  ??????doc_span????????????token???whole doc tokens????????????
                    # ????????????BERT?????????????????????tok_start_position??????token?????????doc??????token_idx??? doc_offset??????doc???BERT??????????????????
                    # tok_start_position - doc_start ???token?????????span????????????, + offset?????????BERT?????????
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 2:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens_query: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens_query]))
                tf.logging.info("tokens_doc: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens_doc]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids_query: %s" % " ".join([str(x) for x in input_ids_query]))
                tf.logging.info("input_ids_doc: %s" % " ".join([str(x) for x in input_ids_doc]))
                tf.logging.info(
                    "input_mask_query: %s" % " ".join([str(x) for x in input_mask_query]))
                tf.logging.info(
                    "input_mask_doc: %s" % " ".join([str(x) for x in input_mask_doc]))
                tf.logging.info(
                    "segment_ids_query: %s" % " ".join([str(x) for x in segment_ids_query]))
                tf.logging.info(
                    "segment_ids_doc: %s" % " ".join([str(x) for x in segment_ids_doc]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens_doc[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))
                    tf.logging.info(
                        "original answer text: %s" % orig_answer_text)

            if unique_id % 1000 == 0:
                tf.logging.info("Writing example %d " % (unique_id - 1000000000))

            feature = InputFeatures(
                unique_id=unique_id,  # ????????????unique id
                example_index=example_index,  # ??????answer-question-doc??? id????????????????????????doc???span??????example_index
                doc_span_index=doc_span_index,
                tokens_query=tokens_query,
                tokens_doc=tokens_doc,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids_query=input_ids_query,
                input_ids_doc=input_ids_doc,
                input_mask_query=input_mask_query,
                input_mask_doc=input_mask_doc,
                segment_ids_query=segment_ids_query,
                segment_ids_doc=segment_ids_doc,
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
    #   Context: The leader was John Smith (1895-1943).   ????????????????????????????????????????????????????????????token 1895 ?????????????????????????????????token ???
    #   ????????????start_token?????????????????????????????????????????????idx??????token???????????????token???token idx
    #   ???????????????????????????????????????1???????????????(1895-1943)?????????token???????????????token??? (?????????idx??????????????????1895
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
    # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    # input_start????????????????????????????????????????????????????????????token???token_idx
    # input_end??? ????????????????????????token???idx????????????????????????????????????(?????????)?????????????????????token???token_idx
    # ???????????????????????????????????????????????????token???????????????
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):  # ????????????????????????( 1895 - 1943 )???????????????token
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
    # right context will always be the same, of course).  ?????????????????????????????????????????????
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    # doc_spans: ???????????????(??????span?????????????????????offset?????????span?????????)
    # cur_span_index: ???????????????span
    # position: ??????token?????????doc????????????, ????????????????????????token?????????cur_span_index?????????span???????????????????????????????????????span???
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


def input_fn_builder(input_file, seq_length_query, seq_length_doc, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids_query": tf.FixedLenFeature([seq_length_query], tf.int64),
        "input_ids_doc": tf.FixedLenFeature([seq_length_doc], tf.int64),
        "input_mask_query": tf.FixedLenFeature([seq_length_query], tf.int64),
        "input_mask_doc": tf.FixedLenFeature([seq_length_doc], tf.int64),
        "segment_ids_query": tf.FixedLenFeature([seq_length_query], tf.int64),
        "segment_ids_doc": tf.FixedLenFeature([seq_length_doc], tf.int64),
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

    output_shape = (batch_size, seq_length, hidden_size)

    # output_weights = tf.get_variable(
    #     "output_weights", [2, hidden_size],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02))
    #
    # output_bias = tf.get_variable(
    #     "output_bias", [2], initializer=tf.zeros_initializer())
    #
    # final_hidden_matrix = tf.reshape(output_layer,
    #                                  [batch_size * seq_length, hidden_size])
    # logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    # logits = tf.nn.bias_add(logits, output_bias)
    #
    # logits = tf.reshape(logits, [batch_size, seq_length, 2])
    # logits = tf.transpose(logits, [2, 0, 1])  # [2, bs, seq_len]      # each position word_embedding mapped to a value
    #
    # unstacked_logits = tf.unstack(logits, axis=0)
    #
    # (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # return (start_logits, end_logits)         # [bs, seq_len]
    return output_layer, output_shape, model


def pool_query_output(output_layer, input_mask, query_model):
    # output_layer: [bs, seq_length, emb_dim]
    if FLAGS.pooling_strategy == "cls":
        tf.logging.info("use cls embedding")
        output_layer = query_model.get_pooled_output()

    elif FLAGS.pooling_strategy == "mean":
        tf.logging.info("use mean embedding")
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
        input_ids_query = features["input_ids_query"]
        input_ids_doc = features["input_ids_doc"]
        input_mask_query = features["input_mask_query"]
        input_mask_doc = features["input_mask_doc"]
        segment_ids_query = features["segment_ids_query"]
        segment_ids_doc = features["segment_ids_doc"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (output_layer_query, output_shape_query, model_query) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids_query,
            input_mask=input_mask_query,
            segment_ids=segment_ids_query,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (output_layer_doc, output_shape_doc, model_doc) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids_doc,
            input_mask=input_mask_doc,
            segment_ids=segment_ids_doc,
            use_one_hot_embeddings=use_one_hot_embeddings)

        def attention_score(q, k):
            # q [B, S, num_label, H], v [B, T, num_label, H]
            q = tf.transpose(q, perm=[0, 2, 1, 3])
            k = tf.transpose(k, perm=[0, 2, 1, 3])
            attention_scores = tf.matmul(q, k, transpose_b=True)
            # attention_scores [B, num_label, S, T]
            attention_scores = tf.multiply(attention_scores,
                                           1.0 / math.sqrt(float(bert_config.hidden_size)))
            attention_scores = tf.reduce_max(attention_scores, axis=1)  # [B, num_label, T]
            return attention_scores

        batch_size, seq_length_doc, hidden_size = output_shape_doc

        doc_embedding = output_layer_doc
        query_embedding = output_layer_query

        query_embedding = tf.layers.dense(query_embedding, units=FLAGS.colbert_dim)
        doc_embedding = tf.layers.dense(doc_embedding, units=FLAGS.colbert_dim)
        B, S, H = modeling.get_shape_list(query_embedding)
        _, T, H = modeling.get_shape_list(doc_embedding)

        transform_weights = tf.get_variable(
            "output_weights", [2 * FLAGS.colbert_dim, FLAGS.colbert_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        query_embedding = tf.reshape(tf.matmul(query_embedding, transform_weights, transpose_b=True), [B, S, 2, H])
        doc_embedding = tf.reshape(tf.matmul(doc_embedding, transform_weights, transpose_b=True), [B, T, 2, H])

        query_embedding, _ = tf.linalg.normalize(query_embedding, ord=2, axis=-1)
        doc_embedding, _ = tf.linalg.normalize(doc_embedding, ord=2, axis=-1)

        query_mask = tf.expand_dims(input_mask_query, axis=-1)
        query_mask = tf.expand_dims(query_mask, axis=-1)
        query_mask = tf.tile(query_mask, tf.constant([1, 1, 2, FLAGS.colbert_dim]))
        query_mask = tf.cast(query_mask, dtype=tf.float32)
        query_embedding = tf.multiply(query_mask, query_embedding)

        doc_mask = tf.expand_dims(input_mask_doc, axis=-1)
        doc_mask = tf.expand_dims(doc_mask, axis=-1)
        doc_mask = tf.tile(doc_mask, tf.constant([1, 1, 2, FLAGS.colbert_dim]))
        doc_embedding = tf.multiply(tf.cast(doc_mask, dtype=tf.float32), doc_embedding)

        score = attention_score(query_embedding, doc_embedding)
        logits = tf.transpose(score, [1, 0, 2])  # [2, bs, seq_len]      # each position word_embedding mapped to a value

        unstacked_logits = tf.unstack(logits, axis=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        # pooled_output_layer_query = pool_query_output(output_layer_query, input_mask_query, model_query)
        # start_logits, end_logits = get_logits(pooled_output_layer_query, output_layer_doc, output_shape_doc)

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
            seq_length_doc = modeling.get_shape_list(input_ids_doc)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length_doc, dtype=tf.float32)
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
                "start_logits": start_logits,  # ?????????????????????doc??????????????????idx
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_logits(pooled_output_layer_query, output_layer_doc, output_shape_doc):
    """
  pooled_output_layer_query: [bs, emb_dim]
  output_layer_doc : [bs, seq_length_doc, emb_dim]
  positions: [bs]
  """
    batch_size, seq_length_doc, hidden_size = output_shape_doc

    pooled_output_layer_query = tf.expand_dims(pooled_output_layer_query, axis=1)  # [bs, 1, emb_dim]
    pooled_output_layer_query = tf.tile(pooled_output_layer_query, [1, seq_length_doc, 1])
    sub_embedding = tf.abs(pooled_output_layer_query - output_layer_doc)
    max_embedding = tf.square(tf.reduce_max([pooled_output_layer_query, output_layer_doc], axis=0))
    regular_embedding = tf.concat([pooled_output_layer_query, output_layer_doc, sub_embedding, max_embedding], -1)
    # #[bs, seq_length_doc, emb_dim]

    output_weights = tf.get_variable(
        "output_weights", [2, hidden_size * 4],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(regular_embedding,
                                     [batch_size * seq_length_doc, hidden_size * 4])  # regular_embedding?????????????????????
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length_doc, 2])
    logits = tf.transpose(logits, [2, 0, 1])  # [2, bs, seq_len]      # each position word_embedding mapped to a value

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
    return (start_logits, end_logits)  # [bs, seq_len]


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
        example_index_to_features[feature.example_index].append(feature)  # ??????????????????doc?????????span????????????example_index

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result  # ????????????span?????????id????????????????????????

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]  # ??????doc?????????span?????????

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):  # ??????doc?????????span
            result = unique_id_to_result[feature.unique_id]  # ???span???????????????
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)  # ???span??????start index????????????n_best_size
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
                    # We could hypothetically create invalid predictions, e.g., predict # ?????????doc?????????span?????????span?????????n_best_size X n_best_size ??????????????????????????????
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens_doc):
                        continue
                    if end_index >= len(feature.tokens_doc):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        # {??????doc??????token??????????????????BERT????????????idx: token??????????????????doc????????????idx}, ?????????????????????start_index??????doc span???
                        # token_to_orig_map ??????key?????????????????????token?????????????????????padding
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):  # start???????????????????????????
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,  # ???0??????, ??????doc????????????span???idx
                            start_index=start_index,  # ?????????????????????n_best_size???????????????start_token???BERT????????????index
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],  # ?????????
                            end_logit=result.end_logits[end_index]))
                    # ???????????????????????? n_best * n_best????????????????????????
        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,  # ???0???0??????????????????????????????span??????doc????????????
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))  # ??????doc???span??????????????????????????????????????????
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),  # ??????doc??????????????????span????????????logit??????????????????
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:  # ??????doc?????????????????????n_span * n_best * n_best???????????????????????????????????????n_best_size???span??????
                break
            feature = features[pred.feature_index]  # ???????????????feature
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens_doc[pred.start_index:(pred.end_index + 1)]  # ???????????????????????????, ?????????span????????????
                orig_doc_start = feature.token_to_orig_map[
                    pred.start_index]  # ??????  {??????doc??????token??????????????????BERT????????????idx: token??????????????????doc????????????idx}
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[
                              orig_doc_start:(orig_doc_end + 1)]  # ??????????????????, ?????????????????????????????????, ???: ?????? (1893--1902)
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)  # ?????????????????????????????????????????????????????????, ???????????????????????????, ???: ?????? (1893--1902)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,  # ??????doc?????????????????????span?????????????????????,
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

        total_scores = []  # ?????????????????????span???????????????????????????
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)  # ??????????????? n_best???????????????????????????softmax

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:  # ?????????null, ??????????????????????????????
            all_predictions[example.qas_id] = nbest_json[0]["text"]  # ????????????????????????id
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
        features["input_ids_query"] = create_int_feature(feature.input_ids_query)
        features["input_ids_doc"] = create_int_feature(feature.input_ids_doc)
        features["input_mask_query"] = create_int_feature(feature.input_mask_query)
        features["input_mask_doc"] = create_int_feature(feature.input_mask_doc)
        features["segment_ids_query"] = create_int_feature(feature.segment_ids_query)
        features["segment_ids_doc"] = create_int_feature(feature.segment_ids_doc)

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
        return re.sub(r'\b(a|an|the)\b', ' ', text)  # ???????????????????????????

    def white_space_fix(text):
        return ' '.join(text.split())  # ???????????????????????????

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)  # ??????????????????

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
               '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '??????', '??', '???',
               '???', '???', '???', '???', '???', '???', '???', '???']
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
        # ????????????jsonl?????????????????????
        # ????????????
        # {"id":, "seq1": (question), "seq2": (doc), , "label": {"ans": [[start_idx, content]]}}
        # ???????????????????????????????????????????????????
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
            original_file_data.append(line_data)  # ??????????????????????????????????????????

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
        keep_checkpoint_max=50,
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

        if not os.path.exists(train_file_tfr):
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
                seq_length_query=FLAGS.max_query_length + 2,  # [CLS], query, <PAD>, [SEP]
                seq_length_doc=FLAGS.max_doc_length + 2,  # [CLS], doc, <PAD>, [SEP]
                is_training=True,
                drop_remainder=True)
        else:
            train_input_fn = input_fn_builder(
                input_file=train_file_tfr,
                seq_length_query=FLAGS.max_query_length + 2,  # [CLS], query, <PAD>, [SEP]
                seq_length_doc=FLAGS.max_doc_length + 2,  # [CLS], doc, <PAD>, [SEP]
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
            seq_length_query=FLAGS.max_query_length + 2,  # [CLS], query, <PAD>, [SEP]
            seq_length_doc=FLAGS.max_doc_length + 2,  # [CLS], doc, <PAD>, [SEP]
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

        best_f1, best_em, best_ckpt_f1, best_ckpt_em = 0, 0, 0, 0
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
                tf.logging.info("evaluating {}...".format(filename))
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
                f1, em = float(eval_result['F1']), float(eval_result['EM'])
                if f1 > best_f1:
                    best_f1 = f1
                    best_ckpt_f1 = filename
                if em > best_em:
                    best_em = em
                    best_ckpt_em = filename

                tf.logging.info("***** Eval results %s *****" % (filename))
                writer.write("***** Eval results %s *****\n" % (filename))
                for key in sorted(eval_result.keys()):
                    tf.logging.info("  %s = %s", key, str(eval_result[key]))
                    writer.write("%s = %s\n" % (key, str(eval_result[key])))

        tf.logging.info("  best f1: {} from {}".format(best_f1, best_ckpt_f1))
        tf.logging.info("  best em: {} from {}".format(best_em, best_ckpt_em))

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
            seq_length_query=FLAGS.max_query_length + 2,  # [CLS], query, <PAD>, [SEP]
            seq_length_doc=FLAGS.max_doc_length + 2,  # [CLS], doc, <PAD>, [SEP]
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
