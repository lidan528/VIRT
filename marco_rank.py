import tensorflow as tf
import pickle
import numpy as np
import sys


def init_flags():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Launch Distributed Training")
    parser.add_argument("--doc_emb_file", type=str, default=None)
    parser.add_argument("--query_emb_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=1000)
    return parser


def read_embedding(emb_file):
    with tf.io.gfile.GFile(emb_file, 'rb') as f:
        try:
            while True:
                ex = pickle.load(f)
                yield ex
        except EOFError:
            raise StopIteration


def read_query_embedding(emb_file):
    query_ids = []
    query_emb = []
    for ex in read_embedding(emb_file):
        query_ids.append(ex['query_guid'])
        query_emb.append(ex['query_embedding'])
    query_emb = np.array(query_emb)
    return query_ids, query_emb


def read_doc_embedding(emb_file, batch_size):
    doc_ids = []
    doc_emb = []
    for ex in read_embedding(emb_file):
        if len(doc_ids) < batch_size:
            doc_emb.append(ex['doc_embedding'])
            doc_ids.append(ex['doc_guid'])
        else:
            doc_emb = np.array(doc_emb)
            doc_ids = np.array(doc_ids)
            yield doc_ids, doc_emb
            doc_emb = []
            doc_ids = []
    if len(doc_ids) > 0:
        doc_emb = np.array(doc_emb)
        doc_ids = np.array(doc_ids)
        yield doc_ids, doc_emb


def rank_doc(query_emb_file, doc_emb_files, flags):
    all_top_doc_scores = None
    all_top_doc_ids = None
    query_ids, query_emb = read_query_embedding(query_emb_file)
    processed_doc_num = 0
    for doc_emb_file in doc_emb_files:
        for doc_ids, doc_emb in read_doc_embedding(doc_emb_file, flags.batch_size):
            doc_scores = np.matmul(query_emb, doc_emb.T)
            processed_doc_num += len(doc_emb)
            if processed_doc_num % flags.batch_size == 0:
                tf.logging.info(f"processed {processed_doc_num} documents.")
            if all_top_doc_ids is None:
                top_doc_indices = np.argsort(doc_scores, axis=-1)[:, :flags.topk]
                all_top_doc_ids = doc_ids[top_doc_indices]
                all_top_doc_scores = doc_scores[np.arange(len(doc_scores))[:, np.newaxis], top_doc_indices]
            else:
                all_top_doc_ids = np.concatenate([all_top_doc_ids,
                                                  np.tile(doc_ids, (len(all_top_doc_ids), 1))], axis=-1)
                all_top_doc_scores = np.concatenate([all_top_doc_scores, doc_scores], axis=-1)
                top_doc_indices = np.argsort(all_top_doc_scores, axis=-1)[:, :flags.topk]
                all_top_doc_ids = all_top_doc_ids[np.arange(len(all_top_doc_ids))[:, np.newaxis], top_doc_indices]
                all_top_doc_scores = all_top_doc_scores[np.arange(len(all_top_doc_scores))[:, np.newaxis], top_doc_indices]

    return all_top_doc_ids, query_ids


def process(flags):
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
        file_name_output = "rank_" + file_name_input

        print("start process file:" + file_name_input + " " + "result file:" + file_name_output)
        # from io import FileIO
        # input_file = FileIO(row[1])
        input_file = row[1]
        doc_emb_files = tf.io.gfile.glob(flags.doc_emb_file)
        all_top_doc_ids, query_ids = rank_doc(input_file, doc_emb_files, flags)
        output_file = flags.output_dir + file_name_output

        with tf.io.gfile.GFile(output_file, 'w') as f:
            for query_id, doc_ids in zip(query_ids, all_top_doc_ids):
                for i, doc_id in enumerate(doc_ids):
                    f.write(f"{query_id}\t{doc_id}\t{i+1}\n")

    return _process


def main(argv):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    parser = init_flags()
    flags = parser.parse_args(argv[1:])
    input_dir = flags.query_emb_dir
    tf.logging.set_verbosity(tf.logging.INFO)

    textFiles = sc.wholeTextFiles(input_dir, minPartitions=200, use_unicode=True)
    print(textFiles.map(process(flags)).collect())


if __name__ == '__main__':
    main(sys.argv)