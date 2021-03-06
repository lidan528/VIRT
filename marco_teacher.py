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
    "max_seq_length", None,
    "The maximum total input sequence length after WordPiece tokenization. "
    "In the origin bert model"
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

# tf.flags.DEFINE_integer("poly_first_m", 64, "if use poly-encoder, number of document embeddings to choose")

flags.DEFINE_integer("colbert_dim", 128, "reduction dimension of colbert")


def file_based_input_fn_builder(input_file, seq_length,
                                is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_masks": tf.FixedLenFeature([seq_length], tf.int64),
        "input_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64)
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


def model_fn_builder(bert_config,
                     num_rele_label,
                     init_checkpoint_teacher,
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
            input_ids = features["input_ids"]
            input_segment_ids = features["input_segment_ids"]
            input_masks = features["input_masks"]
            label = features["label"]

        teacher_output_layer, model_teacher = create_model_bert(bert_config=bert_config, is_training=False,
                                                                input_ids=input_ids,
                                                                input_mask=input_masks,
                                                                segment_ids=input_segment_ids,
                                                                use_one_hot_embeddings=use_one_hot_embeddings,
                                                                scope="bert_teacher",
                                                                is_reuse=tf.AUTO_REUSE)
        loss_teacher, per_example_loss_teacher, logits_teacher, probabilities_teacher = \
            get_prediction_teacher(teacher_output_layer=teacher_output_layer,
                                   num_labels=num_rele_label,
                                   labels=label,
                                   is_training=False)

        total_loss = loss_teacher

        vars_teacher = tf.trainable_variables()  # stu + teacher

        # vars_teacher: bert_structure: 'bert_teacher/...',  cls_structure: 'cls_teacher/..'
        # params_ckpt_teacher: bert_structure: 'bert/...', cls_structure: '...'
        assignment_map_teacher, initialized_variable_names_teacher = \
            modeling.get_assignment_map_from_checkpoint_teacher(
                vars_teacher, init_checkpoint_teacher
            )
        tf.train.init_from_checkpoint(init_checkpoint_teacher, assignment_map_teacher)

        tf.logging.info('****-------------------------init teacher----------------------*****')
        for v_t in assignment_map_teacher:
            tf.logging.info(
                '**initialize ${}$ in graph with checkpoint params ${}$**'.format(assignment_map_teacher[v_t], v_t))
        tf.logging.info('--------------------------------------------------------------------')

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, vars_teacher)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
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

    mask = broadcast_ones * mask  # [batch_size, seq_length_self, seq_length_another]
    return mask


def create_att_mask_4d(input_mask_1, input_mask_2):
    input_mask_1 = tf.expand_dims(tf.expand_dims(input_mask_1, axis=1), axis=-1)
    input_mask_2 = tf.expand_dims(tf.expand_dims(input_mask_2, axis=0), axis=-2)
    return input_mask_1 * input_mask_2



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

    mask = broadcast_ones * mask  # [batch_size, seq_length_another, seq_length_self]
    return mask


def create_att_mask_for1(input_mask):
    """
    ????????????input_mask????????????seq_len???
    ??????????????????, ?????????input_mask??????i????????????0?????????????????????i????????????0
    ????????????input_mask???0????????????att???????????????att to????????????
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

    size_per_head = int(model_teacher.hidden_size / model_teacher.num_attention_heads)
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
        query_doc_qk = tf.matmul(sbert_query_qw[:, :, 1:-1, :], sbert_doc_kw[:, :, 1:-1, :],
                                 transpose_b=True)  # [bs, num_heads, 128, 128]
        query_doc_qk = tf.multiply(query_doc_qk,
                                   1.0 / math.sqrt(float(size_per_head)))
        query_doc_att_matrix_mask = create_att_mask(input_mask_sbert_doc,
                                                    FLAGS.max_seq_length_query)  # doc??????padding??????????????????attend, [bs, 130, 130]
        query_doc_att_matrix_mask = tf.expand_dims(query_doc_att_matrix_mask[:, 1:-1, 1:-1],
                                                   axis=[1])  # to [bs, 1, seq_len=128, seq_len=128]
        query_doc_att_matrix_mask_adder = (1.0 - tf.cast(query_doc_att_matrix_mask, tf.float32)) * -10000.0
        query_doc_att_scores = query_doc_qk + query_doc_att_matrix_mask_adder
        query_doc_att_probs = tf.nn.softmax(query_doc_att_scores)

        sbert_att_shape = modeling.get_shape_list(sbert_query_qw, expected_rank=4)  # [bs, num_heads, seq_len, head_dim]
        seq_len_sbert = sbert_att_shape[2]
        bert_att_score_query_doc = bert_att_score[:, :, 1:(seq_len_sbert - 1),
                                   seq_len_sbert:-1]  # [bs, num_heads, seq_len=128, seq_len=128]
        bert_att_score_query_doc = bert_att_score_query_doc + query_doc_att_matrix_mask_adder
        bert_att_probs_query_doc = tf.nn.softmax(bert_att_score_query_doc)
        loss = loss + tf.losses.mean_squared_error(query_doc_att_probs, bert_att_probs_query_doc)
        # -------------------------------------------------------------------

        doc_query_qk = tf.matmul(sbert_doc_qw[:, :, 1:-1, :], sbert_query_kw[:, :, 1:-1, :],
                                 transpose_b=True)  # [bs, num_heads, 128, 128]
        doc_query_qk = tf.multiply(doc_query_qk,
                                   1.0 / math.sqrt(float(size_per_head)))
        doc_query_att_matrix_mask = create_att_mask(input_mask_sbert_query, FLAGS.max_seq_length_doc)
        doc_query_att_matrix_mask = tf.expand_dims(doc_query_att_matrix_mask[:, 1:-1, 1:-1],
                                                   axis=[1])  # to [bs, 1, seq_len=128, seq_len=128]
        doc_query_att_matrix_mask_adder = (1.0 - tf.cast(doc_query_att_matrix_mask, tf.float32)) * -10000.0
        doc_query_att_scores = doc_query_qk + doc_query_att_matrix_mask_adder
        doc_query_att_probs = tf.nn.softmax(doc_query_att_scores)

        bert_att_score_doc_query = bert_att_score[:, :, seq_len_sbert:-1, 1:(seq_len_sbert - 1)]
        bert_att_score_doc_query = bert_att_score_doc_query + doc_query_att_matrix_mask_adder
        bert_att_probs_doc_query = tf.nn.softmax(bert_att_score_doc_query)
        loss = loss + tf.losses.mean_squared_error(doc_query_att_probs, bert_att_probs_doc_query)

        num += 1

    loss = loss / num

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
    return output_layer  # [bs_size, emb_dim]


def get_pooled_embeddings_for2(encoder_layer1, encoder_layer2, input_mask1, input_mask2):
    """
    ????????????layer?????????mean pool?????????????????????input???padding???(mask???0)????????????
    encoder_layer:  [bs, seq_len, emb_dim]
    input_mask: [bs, seq_len]
    """
    input_mask = tf.concat([input_mask1, input_mask2], axis=-1)  # [bs, seq_len1+seq_len2]
    encoder_layer = tf.concat([encoder_layer1, encoder_layer2], axis=1)  # [bs, seq_len1+seq_len2, emb_dim]
    output_layer = get_pooled_embeddings(encoder_layer, input_mask)
    return output_layer


def cosine_distance(X1, X2):
    """
    ???????????????
    X1 : [bs, emb_dim]
    X2:  [bs, emb_dim]
    """
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=-1))  # [bs]
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))  # [bs]
    X1_norm_X2_norm = X1_norm * X2_norm

    X1_X2 = X1 * X2  # [bs * emb_dim]
    X1_X2 = tf.reduce_sum(X1_X2, axis=-1)  # [bs]

    cosine = X1_X2 / X1_norm_X2_norm  # ????????????????????????, [bs]
    cosine = tf.reduce_mean(cosine, axis=-1)
    return 1 - cosine  # distance


def get_pooled_loss(teacher_model, student_model_query, student_model_doc,
                    input_mask_teacher, input_mask_query, input_mask_doc,
                    mode):
    """
    ??????pooled???representation???????????????????????????
    teacher??????mean pooling
    mode1: [direct_mean] student???????????????mean pooling????????????teacher???mean pooling???????????????
    mode2: [map_mean] student?????????query???doc??????mean pooling, ????????????|v1-v2, v1+v2...|??????????????????teacher ??????
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
            pooled_teacher_layer = get_pooled_embeddings(teacher_layer, input_mask_teacher)  # [bs, emb_dim]
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
            pooled_teacher_layer = get_pooled_embeddings(teacher_layer, input_mask_teacher)  # [bs, emb_dim]
            pooled_query_layer = get_pooled_embeddings(query_layer, input_mask_query)  # [bs, emb_dim]
            pooled_doc_layer = get_pooled_embeddings(doc_layer, input_mask_doc)  # [bs, emb_dim]
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
        assert 1 == 2


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


def in_batch_negative_loss(query_embedding, doc_embedding, label_ids, num_docs):
    # select one batch query_embedding
    query_embedding_in_batch = query_embedding[::num_docs, :]
    # [bs, bs*num_docs]
    scores_in_batch = tf.matmul(query_embedding_in_batch, doc_embedding, transpose_b=True)
    # in-batch label_ids [bs, bs*num_docs]
    label_ids = tf.cast(label_ids, dtype=tf.float32)
    label_ids_in_batch = tf.eye(FLAGS.train_batch_size * num_docs) * tf.expand_dims(label_ids,
                                                                                    axis=0)  # [bs*num_docs, bs*num_docs]
    label_ids_in_batch = label_ids_in_batch[::num_docs, :]  # [bs, bs*num_docs]
    scores_in_batch = tf.nn.softmax(scores_in_batch, axis=1)  # [bs, bs*num_docs]
    log_scores_in_batch = tf.log(scores_in_batch + 1e-7)
    loss_in_batch = - tf.reduce_mean(tf.reduce_sum(label_ids_in_batch * log_scores_in_batch, axis=-1))
    return loss_in_batch, scores_in_batch, label_ids_in_batch


def contrastive_loss_self(teacher_model, query_model, doc_model,
                          input_mask_teacher, input_mask_query, input_mask_doc,
                          truth_labels):
    """
    ??????????????????????????????????????????????????????????????? S1 S2???
    ??????S1???S2??????????????????????????????????????????????????????batch?????????????????????????????????????????????
    ????????????????????????
    """

    def cos_loss_self(matrix_a, matrix_b, label_mask):
        """
        label_mask: ?????????????????????????????????????????????, label???1  [bs]
        """
        dot_result = tf.matmul(matrix_a, matrix_b, transpose_b=True)  # [bs , bs]   dot
        norm2_a_output = tf.sqrt(tf.reduce_sum(tf.square(matrix_a), axis=1, keep_dims=True))  # [bs, 1]  |a|
        norm2_b_output = tf.sqrt(tf.reduce_sum(tf.square(matrix_b), axis=1, keep_dims=True))  # [bs, 1], |b|
        norm2_ab = tf.matmul(norm2_a_output, norm2_b_output, transpose_b=True)  # [bs, bs], |a||b|
        cos_sim = tf.divide(dot_result, norm2_ab)  # batch_size * batch_size
        cos_sim = tf.nn.softmax(cos_sim, axis=1)
        log_cos_elem = tf.log(cos_sim)
        diag_elem = tf.multiply(tf.eye(tf.shape(cos_sim)[0]), log_cos_elem)  # [bs, bs] only diag remained
        label_mask = tf.cast(tf.expand_dims(label_mask, axis=-1), dtype=tf.float32)  # [bs, 1]
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

    return -1 * loss / cnt


def contrastive_loss_teacher_separately(teacher_model, query_model, doc_model,
                                        input_mask_teacher, input_mask_query, input_mask_doc,
                                        truth_labels):
    """
    teacher?????????query??????????????????batch??????query????????????;  ???????????????
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

        query_length = modeling.get_shape_list(query_layer, expected_rank=3)[1]  # [bs, query_len, emb_dim]
        # doc_length = modeling.get_shape_list(doc_layer, expected_rank=3)[1]
        teacher_query = get_pooled_embeddings(teacher_layer[:, 1:query_length - 1, :],
                                              input_mask_teacher[:, 1:query_length - 1])
        teacher_doc = get_pooled_embeddings(teacher_layer[:, query_length:-1, :],
                                            input_mask_teacher[:, query_length:-1])

        loss_query = cos_sim_loss_for_contrast(teacher_query, pooled_query_layer)
        loss_doc = cos_sim_loss_for_contrast(teacher_doc, pooled_doc_layer)
        loss += (loss_query + loss_doc) / 2.0
        cnt += 1
    return -1 * loss / cnt


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

    label_list = [0, 1]

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
        keep_checkpoint_max=80,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_steps = int(FLAGS.num_train_steps)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_rele_label=len(label_list),
        init_checkpoint_teacher=FLAGS.init_checkpoint_teacher,
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
        train_file = tf.io.gfile.glob(FLAGS.train_data_dir)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_file = tf.io.gfile.glob(FLAGS.eval_data_dir)

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

        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

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



if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    # flags.mark_flag_as_required("max_seq_length_bert")
    # flags.mark_flag_as_required("max_seq_length_sbert")
    flags.mark_flag_as_required("pooling_strategy")
    flags.mark_flag_as_required("init_checkpoint_teacher")
    tf.app.run()
