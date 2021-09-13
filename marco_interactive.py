#! usr/bin/env python3

# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import sklearn.metrics

import modeling, tokenization,optimization
import collections
from utils.configurable import JsonConfigurable

config = JsonConfigurable({
    "model":{
        "max_query_length": 16,
        "max_doc_length": 128,
        "vocab_file": "/home/hadoop-aipnlp/cephfs/data/bujiahao/bert-base/vocab.txt",
        "bert_config_file": "/home/hadoop-aipnlp/cephfs/data/bujiahao/bert-base/bert_config.json",
        "init_checkpoint": "/home/hadoop-aipnlp/cephfs/data/tanghongyin/workspace/BERT-dual/output/"
                           "commodity_relevance_binary_classify_sentence_bert_cosine_transform_reuse_128/model.ckpt-2000",
        "do_lower_case": True,
        "share_tower": True,
        "pooling_strategy": "mean",
        "score_function": "cosine",
        "use_em_feature": False,
        "output_dim": -1
    }
})

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids_a, input_mask_a, segment_ids_a,input_ids_b, input_mask_b, segment_ids_b):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a

        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b


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
    return output_layer, model


class ModelIOHelper:
    def __init__(self):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=config.model.vocab_file, do_lower_case=config.model.do_lower_case)

    def create_query_example(self, query_text):
        query_text = tokenization.convert_to_unicode(query_text)
        query_tokens = self.tokenizer.tokenize(query_text)
        if len(query_tokens) > config.model.max_query_length - 2:
            query_tokens = query_tokens[:config.model.max_query_length - 2]
        query_tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
        query_input_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
        query_segment_ids = [0] * len(query_input_ids)
        query_mask = [1] * len(query_input_ids)
        while len(query_input_ids) < config.model.max_query_length:
            query_input_ids.append(0)
            query_segment_ids.append(0)
            query_mask.append(0)
        assert len(query_input_ids) == config.model.max_query_length
        assert len(query_segment_ids) == config.model.max_query_length
        assert len(query_mask) == config.model.max_query_length
        return query_tokens, query_input_ids, query_segment_ids, query_mask

    def create_doc_example(self, spu_name_text):
        doc_text = tokenization.convert_to_unicode(spu_name_text)
        doc_tokens = self.tokenizer.tokenize(doc_text)
        if len(doc_tokens) > config.model.max_doc_length - 2:
            doc_tokens = doc_tokens[:config.model.max_doc_length - 2]
        doc_tokens = ["[CLS]"] + doc_tokens + ["[SEP]"]
        doc_input_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
        doc_segment_ids = [1] * len(doc_input_ids)
        doc_mask = [1] * len(doc_input_ids)
        while len(doc_input_ids) < config.model.max_doc_length:
            doc_input_ids.append(0)
            doc_segment_ids.append(0)
            doc_mask.append(0)
        assert len(doc_input_ids) == config.model.max_doc_length
        assert len(doc_segment_ids) == config.model.max_doc_length
        assert len(doc_mask) == config.model.max_doc_length
        return doc_tokens, doc_input_ids, doc_segment_ids, doc_mask


def create_model(max_seq_length):
    bert_config = modeling.BertConfig.from_json_file(config.model.bert_config_file)

    input_ids = tf.placeholder(
        shape=[None, max_seq_length],
        dtype=tf.int32,
        name="input_ids"
    )
    input_segment_ids = tf.placeholder(
        shape=[None, max_seq_length],
        dtype=tf.int32,
        name="input_segment_ids"
    )
    input_mask = tf.placeholder(
        shape=[None, max_seq_length],
        dtype=tf.int32,
        name="input_mask"
    )
    output_layer, model = create_model_sbert(
        bert_config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=input_segment_ids,
        use_one_hot_embeddings=False,
        scope="bert_student",
        is_reuse=tf.AUTO_REUSE,
        pooling=True
    )

    return output_layer, input_ids, input_segment_ids, input_mask


def cosine_similarity(query_embedding, doc_embedding):
    # cosine similarity
    doc_score = tf.reduce_sum(query_embedding * doc_embedding, axis=-1)
    query_norm = tf.norm(query_embedding, ord='euclidean', axis=-1)
    doc_norm = tf.norm(doc_embedding, ord='euclidean', axis=-1)
    qd_norm = query_norm * doc_norm
    doc_score = doc_score / qd_norm
    return doc_score


def init_from_student_checkpoint(vars_student, init_checkpoint_student):
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
