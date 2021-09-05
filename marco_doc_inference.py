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

import numpy

numpy.set_printoptions(threshold=sys.maxsize)

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
    "train_data_dir", None,
    "The input data path. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "eval_data_dir", None,
    "The input data path. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "predict_data_dir", None,
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
    "use_layer_distill", False,
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

flags.DEFINE_integer("save_checkpoints_steps", 10000,
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

flags.DEFINE_bool("use_resnet_predict", False, "Whether to use resnet in predict.")

flags.DEFINE_bool("use_in_batch_neg", False, "Whether use in-batch negatives.")

flags.DEFINE_integer("poly_first_m", 64, "number of tokens to ue in poly-encoder.")

flags.DEFINE_integer("num_negatives_in_tfr", 4, "number of negative samples.")

flags.DEFINE_integer("num_negatives", 4, "number of negative samples.")

flags.DEFINE_integer("num_train_steps", 5000000, "number of train steps.")

tf.flags.DEFINE_string(
    "model_type", None,
    "which model to use"
)

flags.DEFINE_string("output_filename", None, "")

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


def file_based_input_fn_builder(input_file, seq_length_doc,
                                is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "doc_guid": tf.FixedLenFeature([], tf.int64),
        "doc_ids": tf.FixedLenFeature([seq_length_doc], tf.int64),
        "doc_masks": tf.FixedLenFeature([seq_length_doc], tf.int64),
        "doc_segment_ids": tf.FixedLenFeature([seq_length_doc], tf.int64)
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

        # output_layer: [bs_size, max_len, emb_dim];        input_mask: [bs_size, max_len]
        mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
        masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
        sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
        actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
        actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
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

            # output_layer: [bs_size, max_len, emb_dim];        input_mask: [bs_size, max_len]
            mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)  # mask: [bs_size, max_len, 1]
            masked_output_layer = mask * output_layer  # [bs_size, max_len, emb_dim]
            sum_masked_output_layer = tf.reduce_sum(masked_output_layer, axis=1)  # [bs_size, emb_dim]
            actual_token_nums = tf.reduce_sum(input_mask, axis=-1)  # [bs_size]
            actual_token_nums = tf.cast(tf.expand_dims(actual_token_nums, axis=-1), dtype=tf.float32)  # [bs_size, 1]
            output_layer = sum_masked_output_layer / actual_token_nums

            # token_embedding_sum = tf.reduce_sum(output_layer, 1)  # batch*hidden_size
            # output_layer = token_embedding_sum/length
            # output_layer = token_embedding_sum / FLAGS.max_seq_length
        else:
            tf.logging.info("pooling_strategy error")
            assert 1 == 2

    return output_layer, model


def model_fn_builder(bert_config,
                     init_checkpoint_student,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        doc_ids = features["doc_ids"]
        doc_segment_ids = features["doc_segment_ids"]
        doc_masks = features["doc_masks"]
        doc_guid = features["doc_guid"]

        input_ids_sbert_b = doc_ids
        input_mask_sbert_b = doc_masks
        segment_ids_sbert_b = doc_segment_ids

        if FLAGS.model_type == 'poly':
            tf.logging.info("*********** use poly encoder as the model backbone...*******************")
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=True)

        elif FLAGS.model_type == 'late_fusion':
            tf.logging.info("*********** use late fusion as the model backbone...*******************")
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=False)

        else:
            doc_embedding, model_stu_doc = create_model_sbert(bert_config=bert_config, is_training=is_training,
                                                              input_ids=input_ids_sbert_b,
                                                              input_mask=input_mask_sbert_b,
                                                              segment_ids=segment_ids_sbert_b,
                                                              use_one_hot_embeddings=use_one_hot_embeddings,
                                                              scope="bert_student",
                                                              is_reuse=tf.AUTO_REUSE,
                                                              pooling=True)

        vars_student = tf.trainable_variables()  # bert_structure: 'bert_student/...',  cls_structure: 'cls_student/..'

        assignment_map_student, initialized_variable_names_student = \
            modeling.get_assignment_map_from_checkpoint_student(
                vars_student, init_checkpoint_student
            )
        tf.train.init_from_checkpoint(init_checkpoint_student, assignment_map_student)

        tf.logging.info('****-------------------------init student----------------------*****')
        for v_s in assignment_map_student:
            tf.logging.info(
                '**initialize ${}$ in graph with checkpoint params ${}$**'.format(assignment_map_student[v_s], v_s))
        tf.logging.info('--------------------------------------------------------------------')

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"doc_embedding": doc_embedding, "doc_guid": doc_guid},
            scaffold_fn=None)

        return output_spec

    return model_fn


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


def create_att_mask_for2(input_mask_1, input_mask_2):
    """
    相当于将input_mask列表复制seq_len次
    得到的矩阵中, 如果原input_mask的第i个元素为0，那么矩阵的第i列就全为0
    从而防止input_mask为0的元素被att，但它可以att to别的元素
    如果输入的是mask_doc，那么就要复制query_length次，形成[query_length , mask_doc]的mask
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

    mask = broadcast_ones * mask  # [batch_size, seq_length_self, seq_length_another]
    return mask


def create_att_mask_4d(input_mask_1, input_mask_2):
    input_mask_1 = tf.expand_dims(tf.expand_dims(input_mask_1, axis=1), axis=-1)
    input_mask_2 = tf.expand_dims(tf.expand_dims(input_mask_2, axis=0), axis=-2)
    return input_mask_1 * input_mask_2


def create_att_mask(input_mask, seq_length_another):
    """
    相当于将input_mask列表复制seq_len次
    得到的矩阵中, 如果原input_mask的第i个元素为0，那么矩阵的第i列就全为0
    从而防止input_mask为0的元素被att，但它可以att to别的元素
    如果输入的是mask_doc，那么就要复制query_length次，形成[query_length , mask_doc]的mask
    """
    to_shape = modeling.get_shape_list(input_mask, expected_rank=2)  # [batch-size, seq_len]
    batch_size = to_shape[0]
    seq_length_self = to_shape[1]
    mask = tf.cast(
        tf.reshape(input_mask, [batch_size, 1, seq_length_self]), tf.float32)
    broadcast_ones = tf.ones(
        shape=[batch_size, seq_length_another, 1], dtype=tf.float32)

    mask = broadcast_ones * mask  # [batch_size, seq_length_another, seq_length_self]
    return mask


def create_att_mask_for1(input_mask):
    """
    相当于将input_mask列表复制seq_len次
    得到的矩阵中, 如果原input_mask的第i个元素为0，那么矩阵的第i列就全为0
    从而防止input_mask为0的元素被att，但它可以att to别的元素
    """
    to_shape = modeling.get_shape_list(input_mask, expected_rank=2)  # [batch-size, seq_len]
    batch_size = to_shape[0]
    seq_length = to_shape[1]
    mask = tf.cast(
        tf.reshape(input_mask, [batch_size, 1, seq_length]), tf.float32)
    broadcast_ones = tf.ones(
        shape=[batch_size, seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * mask
    return mask


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
    return output_layer  # [bs_size, emb_dim]


def get_pooled_embeddings_for2(encoder_layer1, encoder_layer2, input_mask1, input_mask2):
    """
    获取两个layer整体的mean pool向量，同时去除input中padding项(mask为0)的影响。
    encoder_layer:  [bs, seq_len, emb_dim]
    input_mask: [bs, seq_len]
    """
    input_mask = tf.concat([input_mask1, input_mask2], axis=-1)  # [bs, seq_len1+seq_len2]
    encoder_layer = tf.concat([encoder_layer1, encoder_layer2], axis=1)  # [bs, seq_len1+seq_len2, emb_dim]
    output_layer = get_pooled_embeddings(encoder_layer, input_mask)
    return output_layer


def cosine_distance(X1, X2):
    """
    余弦相似度
    X1 : [bs, emb_dim]
    X2:  [bs, emb_dim]
    """
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=-1))  # [bs]
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))  # [bs]
    X1_norm_X2_norm = X1_norm * X2_norm

    X1_X2 = X1 * X2  # [bs * emb_dim]
    X1_X2 = tf.reduce_sum(X1_X2, axis=-1)  # [bs]

    cosine = X1_X2 / X1_norm_X2_norm  # 相似度，越大越好, [bs]
    cosine = tf.reduce_mean(cosine, axis=-1)
    return 1 - cosine  # distance


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

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
        keep_checkpoint_max=80,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint_student=FLAGS.init_checkpoint_student,
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

    if FLAGS.do_predict:
        doc_predict_file = tf.io.gfile.glob(FLAGS.predict_data_dir)
        tf.logging.info("***** Running prediction*****")
        predict_input_fn = file_based_input_fn_builder(
            input_file=doc_predict_file,
            seq_length_query=FLAGS.max_seq_length_doc,
            is_training=False,
            drop_remainder=False)

        output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.output_filename)
        past_step = 0

        with tf.gfile.GFile(output_predict_file, "w") as f:
            tf.logging.info("***** Predict results *****")
            for prediction in estimator.predict(input_fn=predict_input_fn):
                past_step += 1
                if past_step % FLAGS.log_step_count_steps == 0:
                    tf.logging.info(f"predict {past_step} instances already.")
                pickle.dump(prediction, f)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("pooling_strategy")
    flags.mark_flag_as_required("init_checkpoint_student")
    tf.app.run()
