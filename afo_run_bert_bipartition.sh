#!/bin/bash
source /opt/meituan/hadoop-gpu/bin/hadoop_user_login_centos7.sh hadoop-aipnlp
source /opt/meituan/tensorflow-release/local_env.sh
/opt/meituan/tensorflow-release/bin/mpi-submit.sh -conf /opt/meituan/lidan65/distill/bert_base_bipartition.xml -files /opt/meituan/lidan65/distill