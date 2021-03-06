from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import numpy as np
# import six
import sys

# reload(sys)
# sys.setdefaultencoding("utf-8")


def init_flags():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Launch Distributed Training")
    parser.add_argument("--input_dir", type=str, help="Input raw text file (or comma-separated list of files).")

    parser.add_argument("--output_dir", type=str, help="Output TF example file (or comma-separated list of files).")

    parser.add_argument("--doc_data", type=str, help="Document data file.")
    parser.add_argument("--query_data", type=str, help="Query data file.")

    parser.add_argument("--vocab_file", type=str, help="The vocabulary file that the BERT model was trained on.")

    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Whether to lower case the input text. "
                             "Should be True for uncased models and False for cased models.")

    parser.add_argument("--max_doc_length", type=int, default=512, help="Maximum input sequence length.")
    parser.add_argument("--max_query_length", type=int, default=128, help="Maximum input sequence length.")

    return parser


def write_example_to_tfr_files(instances, tokenizer, max_query_length, max_doc_length,
                               output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        query_ids = tokenizer.convert_tokens_to_ids(instance.query_tokens)
        query_segment_ids = instance.query_segment_ids
        query_masks = [1] * len(query_ids)
        while len(query_ids) < max_query_length:
            query_ids.append(0)
            query_segment_ids.append(0)
            query_masks.append(0)

        positive_doc_ids = tokenizer.convert_tokens_to_ids(instance.positive_doc_tokens)
        positive_doc_segment_ids = instance.positive_doc_segment_ids
        positive_doc_masks = [1] * len(positive_doc_ids)
        while len(positive_doc_ids) < max_doc_length:
            positive_doc_ids.append(0)
            positive_doc_segment_ids.append(0)
            positive_doc_masks.append(0)

        assert len(positive_doc_ids) == len(positive_doc_segment_ids) == len(positive_doc_masks) == max_doc_length

        negative_doc_ids = tokenizer.convert_tokens_to_ids(instance.negative_doc_tokens)
        negative_doc_segment_ids = instance.negative_doc_segment_ids
        negative_doc_masks = [1] * len(negative_doc_ids)
        while len(negative_doc_ids) < max_doc_length:
            negative_doc_ids.append(0)
            negative_doc_segment_ids.append(0)
            negative_doc_masks.append(0)

        assert len(negative_doc_ids) == len(negative_doc_segment_ids) == len(negative_doc_masks) == max_doc_length

        features = collections.OrderedDict()
        features["query_ids"] = create_int_feature(query_ids)
        features["query_segment_ids"] = create_int_feature(query_segment_ids)
        features["query_masks"] = create_int_feature(query_masks)
        features["positive_doc_ids"] = create_int_feature(positive_doc_ids)
        features["positive_doc_segment_ids"] = create_int_feature(positive_doc_segment_ids)
        features["positive_doc_masks"] = create_int_feature(positive_doc_masks)
        features["negative_doc_ids"] = create_int_feature(negative_doc_ids)
        features["negative_doc_segment_ids"] = create_int_feature(negative_doc_segment_ids)
        features["negative_doc_masks"] = create_int_feature(negative_doc_masks)
        features["label"] = create_int_feature([0])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def write_example_to_tfr_files_lidan528(instances, tokenizer, max_query_length, max_doc_length,
                               output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = collections.OrderedDict()
        features["query_ids"] = create_int_feature(instance.query_ids)
        features["query_segment_ids"] = create_int_feature(instance.query_segment_ids)
        features["query_masks"] = create_int_feature(instance.query_masks)
        features["positive_doc_ids"] = create_int_feature(instance.positive_doc_ids)
        features["positive_doc_segment_ids"] = create_int_feature(instance.positive_doc_segment_ids)
        features["positive_doc_masks"] = create_int_feature(instance.positive_doc_masks)
        features["negative_doc_ids"] = create_int_feature(instance.negative_doc_ids)
        features["negative_doc_segment_ids"] = create_int_feature(instance.negative_doc_segment_ids)
        features["negative_doc_masks"] = create_int_feature(instance.negative_doc_masks)
        features["label"] = create_int_feature([0])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DocExample(object):
    def __init__(self, query_tokens, query_segment_ids,
                 positive_doc_tokens, positive_doc_segment_ids,
                 negative_doc_tokens, negative_doc_segment_ids,
                 query_ids=None, query_masks=None,
                 positive_doc_ids=None, positive_doc_masks=None,
                 negative_doc_ids=None, negative_doc_masks=None):
        self.query_tokens = query_tokens
        self.query_segment_ids = query_segment_ids
        self.positive_doc_tokens = positive_doc_tokens
        self.positive_doc_segment_ids = positive_doc_segment_ids
        self.negative_doc_tokens = negative_doc_tokens
        self.negative_doc_segment_ids = negative_doc_segment_ids
        self.query_ids = query_ids
        self.query_masks = query_masks
        self.positive_doc_ids = positive_doc_ids
        self.positive_doc_masks = positive_doc_masks
        self.negative_doc_ids = negative_doc_ids
        self.negative_doc_masks = negative_doc_masks


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def read_doc_data(doc_file):
    doc_data = {}
    for line in tf.io.gfile.GFile(doc_file):
        doc_id, doc = convert_to_unicode(line).strip().split('\t')
        doc_data[doc_id] = doc
    return doc_data


def read_query_data(query_file):
    query_data = {}
    for line in tf.io.gfile.GFile(query_file):
        query_id, query = convert_to_unicode(line).strip().split('\t')
        query_data[query_id] = query
    return query_data


def _data_generate(doc_data, query_data, id_file):
    line_no = 0
    train_data = []
    # header
    while True:
        line = convert_to_unicode(id_file.readline())
        if not line:
            break
        line_no += 1
        qid, positive_pid, negative_pid = line.strip().split('\t')
        train_data.append({
            "query": query_data[qid],
            "positive_doc": doc_data[positive_pid],
            "negative_doc": doc_data[negative_pid]
        })
    return train_data


def create_examples(train_data, max_query_length, max_doc_length, tokenizer):
    def build_bert_input(tokens_temp, max_seq_length, is_doc=0):
        tokens_p = []
        segment_ids = []

        tokens_p.append("[CLS]")
        segment_ids.append(is_doc)

        for token in tokens_temp:
            tokens_p.append(token)
            segment_ids.append(is_doc)

        tokens_p.append("[SEP]")
        segment_ids.append(is_doc)

        # make sure length not exceed
        if len(tokens_p) > max_seq_length:
            tokens_p = tokens_p[:max_seq_length]
            segment_ids = segment_ids[:max_seq_length]

        return tokens_p, segment_ids

    for ex in train_data:
        query_tokens = tokenizer.tokenize(ex['query'])[:max_query_length - 2]

        positive_doc_tokens = tokenizer.tokenize(ex['positive_doc'])[:max_doc_length - 2]
        negative_doc_tokens = tokenizer.tokenize(ex['negative_doc'])[:max_doc_length - 2]
        query_tokens_p, query_segment_ids = build_bert_input(query_tokens, max_seq_length=max_query_length, is_doc=False)
        positive_doc_tokens_p, positive_doc_segment_ids = build_bert_input(positive_doc_tokens, max_seq_length=max_doc_length, is_doc=True)
        negative_doc_tokens_p, negative_doc_segment_ids = build_bert_input(negative_doc_tokens, max_seq_length=max_doc_length, is_doc=True)

        yield DocExample(query_tokens=query_tokens_p, query_segment_ids=query_segment_ids,
                         positive_doc_tokens=positive_doc_tokens_p, positive_doc_segment_ids=positive_doc_segment_ids,
                         negative_doc_tokens=negative_doc_tokens_p, negative_doc_segment_ids=negative_doc_segment_ids)



def create_examples_lidan528(train_data, max_query_length, max_doc_length, tokenizer):
    def build_bert_input_lidan528(tokens_temp, max_seq_length, is_doc=0):
        tokens_p = []
        segment_ids = []

        tokens_p.append("[CLS]")
        segment_ids.append(is_doc)

        for token in tokens_temp:
            tokens_p.append(token)
            segment_ids.append(is_doc)

        input_ids = tokenizer.convert_tokens_to_ids(tokens_p)  # [CLS], a,a,a
        input_mask = [1] * len(input_ids)  # [CLS], a,a,a

        while len(input_ids) < max_seq_length - 1:  #[CLS], a,a,a, <PAD>
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(is_doc)
        tokens_p.append("[SEP]")
        input_ids += tokenizer.convert_tokens_to_ids(["[SEP]"])  # [CLS], a,a,a,<PAD>, [SEP]
        input_mask.append(1)
        segment_ids.append(is_doc)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return tokens_p, segment_ids, input_ids, input_mask


    for ex in train_data:
        query_tokens = tokenizer.tokenize(ex['query'])[:max_query_length - 2]

        positive_doc_tokens = tokenizer.tokenize(ex['positive_doc'])[:max_doc_length - 2]
        negative_doc_tokens = tokenizer.tokenize(ex['negative_doc'])[:max_doc_length - 2]
        query_tokens_p, query_segment_ids, query_ids, query_masks\
            = build_bert_input_lidan528(query_tokens, max_seq_length=max_query_length, is_doc=False)
        positive_doc_tokens_p, positive_doc_segment_ids, positive_doc_ids, positive_doc_masks\
            = build_bert_input_lidan528(positive_doc_tokens, max_seq_length=max_doc_length, is_doc=True)
        negative_doc_tokens_p, negative_doc_segment_ids, negative_doc_ids, negative_doc_masks\
            = build_bert_input_lidan528(negative_doc_tokens, max_seq_length=max_doc_length, is_doc=True)

        yield DocExample(query_tokens=query_tokens_p, query_segment_ids=query_segment_ids,
                         positive_doc_tokens=positive_doc_tokens_p, positive_doc_segment_ids=positive_doc_segment_ids,
                         negative_doc_tokens=negative_doc_tokens_p, negative_doc_segment_ids=negative_doc_segment_ids,
                         query_ids=query_ids, query_masks=query_masks,
                         positive_doc_ids=positive_doc_ids, positive_doc_masks=positive_doc_masks,
                         negative_doc_ids=negative_doc_ids, negative_doc_masks=negative_doc_masks)


def process(FLAGS, tokenizer):
    # def init_hdfs_env():
    #     import os, commands
    #     if 'HADOOP_HDFS_HOME' in os.environ:
    #         classpath = ''
    #         if 'CLASSPATH' in os.environ:
    #             classpath = os.environ['CLASSPATH']
    #         hadoop_path = os.path.join(os.environ['HADOOP_HDFS_HOME'], 'bin', 'hadoop')
    #         print(">>>>HADOOP PATH is " + hadoop_path)
    #
    #         cmd = """
    #                    {0} classpath --glob
    #                """.format(hadoop_path)
    #
    #         status, hadoop_classpath = commands.getstatusoutput(cmd.strip())
    #         print("CLASSPATH: {0}".format(hadoop_classpath.decode()))
    #         os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath.decode()
    def init_hdfs_env():
        import os
        if 'HADOOP_HDFS_HOME' in os.environ:
            classpath = ''
            if 'CLASSPATH' in os.environ:
                classpath = os.environ['CLASSPATH']
            hadoop_path = os.path.join(os.environ['HADOOP_HDFS_HOME'], 'bin', 'hadoop')
            print(">>>>HADOOP PATH is " + hadoop_path)

            cmd = """
                       {0} classpath --glob
                   """.format(hadoop_path)

            # status, hadoop_classpath = subprocess.getstatusoutput(cmd.strip())

            hadoop_classpath = '/opt/meituan/nodemanager/etc/hadoop:/opt/meituan/nodemanager/share/hadoop/common/lib' \
                               '/jackson-mapper-asl-1.9.13.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/jetty' \
                               '-6.1.26.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/commons-logging-1.1.3' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/hadoop-annotations-2.7.1.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/common/lib/commons-math3-3.1.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/commons-io-2.4.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/httpcore-4.2.5.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/common/lib/snappy-java-1.0.4.1.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/common/lib/commons-compress-1.4.1.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/common/lib/curator-client-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/common/lib' \
                               '/activation-1.1.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/zookeeper-3.4.7' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/servlet-api-2.5.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/tokenservice-0.0.1-20191128.083244-24' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/slf4j-log4j12-1.7.10.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/jsp-api-2.1.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/common/lib/stax-api-1.0-2.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/common/lib/jetty-util-6.1.26.jar:/opt/meituan/nodemanager/share/hadoop/common/lib' \
                               '/avro-1.7.4.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/hadoop-auth-2.7.1' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/curator-recipes-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/commons-codec-1.4.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/commons-httpclient-3.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/commons-digester-1.8.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/jsr305-3.0.0.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/common/lib/httpclient-4.2.5.jar:/opt/meituan/nodemanager/share/hadoop/common' \
                               '/lib/asm-3.2.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/xmlenc-0.52.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/common/lib/jackson-xc-1.9.13.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/jettison-1.1.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/common/lib/commons-net-3.1.jar:/opt/meituan/nodemanager/share/hadoop/common' \
                               '/lib/protobuf-java-2.5.0.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/commons' \
                               '-collections-3.2.1.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/commons-cli-1' \
                               '.2.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/jersey-core-1.9.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/api-util-1.0.0-M20.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/commons-beanutils-1.8.0.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/jersey-json-1.9.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/common/lib/hamcrest-core-1.3.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/common/lib/jackson-jaxrs-1.9.13.jar:/opt/meituan/nodemanager/share/hadoop/common/lib' \
                               '/jaxb-impl-2.2.3-1.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/jets3t-0.9.0' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/htrace-core-3.1.0-incubating' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/apacheds-kerberos-codec-2.0.0' \
                               '-M15.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/netty-3.6.2.Final.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/hadoop-lzo-0.4.20.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/curator-framework-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/junit-4.11.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/common/lib/commons-lang-2.6.jar:/opt/meituan/nodemanager/share/hadoop/common' \
                               '/lib/jersey-server-1.9.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/commons' \
                               '-configuration-1.6.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/jsch-0.1.42' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/jackson-core-asl-1.9.13.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/common/lib/guava-11.0.2.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/api-asn1-api-1.0.0-M20.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/apacheds-i18n-2.0.0-M15.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/jaxb-api-2.2.2.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/common/lib/slf4j-api-1.7.10.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/common/lib/paranamer-2.3.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/xz-1.0' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/common/lib/java-xmlbuilder-0.4.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/common/lib/mockito-all-1.8.5.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/common/lib/log4j-1.2.17.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/common/lib/gson-2.2.4.jar:/opt/meituan/nodemanager/share/hadoop/common/hadoop' \
                               '-common-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/common/hadoop-common-2.7.1' \
                               '-tests.jar:/opt/meituan/nodemanager/share/hadoop/common/hadoop-nfs-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/hdfs:/opt/meituan/nodemanager/share/hadoop/hdfs/lib' \
                               '/jackson-mapper-asl-1.9.13.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/jetty-6' \
                               '.1.26.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/commons-logging-1.1.3.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/hdfs/lib/commons-daemon-1.0.13.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/hdfs/lib/commons-io-2.4.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/xercesImpl-2.9.1.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/hdfs/lib/joda-time-2.9.9.jar:/opt/meituan/nodemanager/share/hadoop/hdfs' \
                               '/lib/servlet-api-2.5.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/metrics-core' \
                               '-3.0.1.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/jetty-util-6.1.26.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/hdfs/lib/jackson-core-2.2.3.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/commons-codec-1.4.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/hdfs/lib/jsr305-3.0.0.jar:/opt/meituan/nodemanager/share/hadoop/hdfs' \
                               '/lib/asm-3.2.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/xmlenc-0.52.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/hdfs/lib/scribe-log4j-shade-1.1.2.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/protobuf-java-2.5.0.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/hdfs/lib/commons-cli-1.2.jar:/opt/meituan/nodemanager/share/hadoop/hdfs' \
                               '/lib/jersey-core-1.9.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/leveldbjni' \
                               '-all-1.8.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/jackson-annotations-2.2.3' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/jetty-util-ajax-9.4.12.v20180830' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/htrace-core-3.1.0-incubating.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/hdfs/lib/commons-lang3-3.7.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/netty-3.6.2.Final.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/hdfs/lib/collect-client-shade-0.0.7.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/hdfs/lib/jackson-databind-2.2.3.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/hdfs/lib/commons-lang-2.6.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/jersey' \
                               '-server-1.9.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/jackson-core-asl-1.9' \
                               '.13.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/lib/guava-11.0.2.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/jetty-util-9.4.12.v20180830.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/netty-all-4.0.23.Final.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/lib/slf4j-api-1.7.10.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/hdfs/lib/xml-apis-1.3.04.jar:/opt/meituan/nodemanager/share/hadoop/hdfs' \
                               '/lib/log4j-1.2.17.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/hadoop-hdfs-rbf-2.7' \
                               '.1.jar:/opt/meituan/nodemanager/share/hadoop/hdfs/hadoop-hdfs-nfs-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/hdfs/hadoop-hdfs-rbf-2.7.1-tests.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/hdfs/hadoop-hdfs-2.7.1-tests.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/hdfs/hadoop-hdfs-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/jackson-mapper-asl-1.9.13.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib' \
                               '/jetty-6.1.26.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons-logging-1.1' \
                               '.3.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons-math3-3.1.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/commons-io-2.4.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/javassist-3.20.0-GA.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/yarn/lib/httpcore-4.2.5.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/snappy-java-1.0.4.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons' \
                               '-compress-1.4.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/curator-client-2.7' \
                               '.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/javax.inject-2.5.0-b05.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/activation-1.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/zookeeper-3.4.7.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/servlet-api-2.5.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib' \
                               '/tokenservice-0.0.1-20191128.083244-24.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/jsp-api-2.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/stax-api-1.0-2' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/javax.ws.rs-api-2.0.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/jetty-util-6.1.26.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/avro-1.7.4.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/curator-recipes-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/commons-codec-1.4.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons' \
                               '-httpclient-3.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons-digester-1' \
                               '.8.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/hk2-api-2.5.0-b05.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/jsr305-3.0.0.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/yarn/lib/httpclient-4.2.5.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/yarn/lib/asm-3.2.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/xmlenc-0.52.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/yarn/lib/jackson-xc-1.9.13.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/jettison-1.1.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/commons-net-3.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib' \
                               '/protobuf-java-2.5.0.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons' \
                               '-collections-3.2.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons-cli-1.2' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/jersey-core-1.9.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/leveldbjni-all-1.8.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/yarn/lib/api-util-1.0.0-M20.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/yarn/lib/commons-beanutils-1.8.0.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib' \
                               '/jersey-json-1.9.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/jackson-jaxrs-1.9' \
                               '.13.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/jaxb-impl-2.2.3-1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/jersey-apache-connector-2.23.2.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/hk2-utils-2.5.0-b05.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/jets3t-0.9.0.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/htrace-core-3.1.0-incubating.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/javax.annotation-api-1.2.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/yarn/lib/osgi-resource-locator-1.0.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/aopalliance-repackaged-2.5.0-b05.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/guice-3.0.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/jersey-client-1.9' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/hk2-locator-2.5.0-b05.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/aopalliance-1.0.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/apacheds-kerberos-codec-2.0.0-M15.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/netty-3.6.2.Final.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/yarn/lib/zookeeper-3.4.7-tests.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/hadoop-lzo-0.4.20.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib' \
                               '/curator-framework-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/javax' \
                               '.inject-1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/commons-lang-2.6.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/yarn/lib/jersey-server-1.9.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/commons-configuration-1.6.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/jsch-0.1.42.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/jackson-core-asl-1.9.13.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/yarn/lib/guava-11.0.2.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/jersey' \
                               '-guice-1.9.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/jersey-common-2.23.2' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/api-asn1-api-1.0.0-M20.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/lib/apacheds-i18n-2.0.0-M15.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/jaxb-api-2.2.2.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/yarn/lib/jersey-guava-2.23.2.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/guice-servlet-3.0.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/paranamer-2' \
                               '.3.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/xz-1.0.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/lib/java-xmlbuilder-0.4.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/yarn/lib/ehcache-3.3.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn' \
                               '/lib/jersey-client-2.23.2.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/log4j-1' \
                               '.2.17.jar:/opt/meituan/nodemanager/share/hadoop/yarn/lib/gson-2.2.4.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/hadoop-yarn-server-nodemanager-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/hadoop-yarn-applications-distributedshell-2.7.1.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-api-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/hadoop-yarn-server-router-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/hadoop-yarn-server-web-proxy-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/hadoop-yarn-server-sharedcachemanager-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-server-common-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-registry-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/yarn/hadoop-yarn-client-2.7.1.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/yarn/hadoop-yarn-server-applicationhistoryservice-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-server-resourcemanager-2.7.1.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-applications-unmanaged-am' \
                               '-launcher-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-common-2.7' \
                               '.1.jar:/opt/meituan/nodemanager/share/hadoop/yarn/hadoop-yarn-server-tests-2.7.1.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/jackson-mapper-asl-1.9.13.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/hadoop-annotations-2.7.1.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/commons-io-2.4.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/mapreduce/lib/snappy-java-1.0.4.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/mapreduce/lib/commons-compress-1.4.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/mapreduce/lib/avro-1.7.4.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/mapreduce/lib/asm-3.2.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib' \
                               '/protobuf-java-2.5.0.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/jersey' \
                               '-core-1.9.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/leveldbjni-all-1.8' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/hamcrest-core-1.3.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/mapreduce/lib/guice-3.0.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/mapreduce/lib/aopalliance-1.0.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/mapreduce/lib/netty-3.6.2.Final.jar:/opt/meituan/nodemanager/share' \
                               '/hadoop/mapreduce/lib/hadoop-lzo-0.4.20.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/mapreduce/lib/junit-4.11.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib' \
                               '/javax.inject-1.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/jersey-server' \
                               '-1.9.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/jackson-core-asl-1.9.13' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/lib/jersey-guice-1.9.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/mapreduce/lib/guice-servlet-3.0.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/mapreduce/lib/paranamer-2.3.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/mapreduce/lib/xz-1.0.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/mapreduce/lib/log4j-1.2.17.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce' \
                               '/hadoop-mapreduce-examples-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce' \
                               '/hadoop-mapreduce-client-shuffle-2.7.1.jar:/opt/meituan/nodemanager/share/hadoop' \
                               '/mapreduce/hadoop-mapreduce-client-jobclient-2.7.1-tests.jar:/opt/meituan/nodemanager' \
                               '/share/hadoop/mapreduce/hadoop-mapreduce-client-hs-plugins-2.7.1.jar:/opt/meituan' \
                               '/nodemanager/share/hadoop/mapreduce/hadoop-mapreduce-client-app-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/mapreduce/hadoop-mapreduce-client-hs-2.7.1.jar:/opt' \
                               '/meituan/nodemanager/share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.7.1.jar' \
                               ':/opt/meituan/nodemanager/share/hadoop/mapreduce/hadoop-mapreduce-client-common-2.7.1' \
                               '.jar:/opt/meituan/nodemanager/share/hadoop/mapreduce/hadoop-mapreduce-client' \
                               '-jobclient-2.7.1.jar:/contrib/capacity-scheduler/*.jar '
            print(f"CLASSPATH: {classpath}")
            print("HADOOP_CLASSPATH: {0}".format(hadoop_classpath))
            os.environ['CLASSPATH'] = classpath + os.pathsep + hadoop_classpath

    def _process(row):
        init_hdfs_env()

        file_name_input = row[0].split('/')[-1]
        file_name_input = file_name_input
        file_name_output = "tfrecord_" + file_name_input

        print("start process file:" + file_name_input + " " + "result file:" + file_name_output)
        from io import StringIO
        # tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        input_file = StringIO(row[1])

        doc_data = read_doc_data(FLAGS.doc_data)
        query_data = read_query_data(FLAGS.query_data)
        train_data = _data_generate(doc_data, query_data, id_file=input_file)

        # examples = create_examples(train_data, FLAGS.max_query_length, FLAGS.max_doc_length, tokenizer)
        examples = create_examples_lidan528(train_data, FLAGS.max_query_length, FLAGS.max_doc_length, tokenizer)

        output_files = [FLAGS.output_dir + file_name_output]
        tf.logging.info("*** Writing to output files ***")
        for output_file in output_files:
            tf.logging.info("  %s", output_file)

        # write_example_to_tfr_files(examples, tokenizer, FLAGS.max_query_length, FLAGS.max_doc_length, output_files)
        write_example_to_tfr_files_lidan528(examples, tokenizer, FLAGS.max_query_length, FLAGS.max_doc_length, output_files)

        print("end process file:" + file_name_input + " " + "result file:" + file_name_output)

    return _process


def main(argv):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    parser = init_flags()
    flags = parser.parse_args(argv[1:])

    input_dir = flags.input_dir
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=flags.vocab_file, do_lower_case=flags.do_lower_case)

    textFiles = sc.wholeTextFiles(input_dir, minPartitions=200, use_unicode=True)
    print(textFiles.map(process(flags, tokenizer)).collect())


if __name__ == "__main__":
    main(sys.argv)