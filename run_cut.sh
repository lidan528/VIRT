nohup: ignoring input
WARNING:tensorflow:Estimator's model_fn (<function model_fn at 0x7fc040e88140>) includes params argument, but params are not passed to Estimator.
INFO:tensorflow:Using config: {'_save_checkpoints_secs': None, '_session_config': None, '_keep_checkpoint_max': 5, '_task_type': 'worker', '_train_distribute': None, '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fc040e80750>, '_model_dir': './lcqmc_mt_cut_l6h384i1200i/', '_save_checkpoints_steps': 1000, '_keep_checkpoint_every_n_hours': 10000, '_service': None, '_num_ps_replicas': 0, '_tpu_config': TPUConfig(iterations_per_loop=1000, num_shards=8, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None), '_tf_random_seed': None, '_save_summary_steps': 100, '_device_fn': None, '_cluster': None, '_num_worker_replicas': 1, '_task_id': 0, '_log_step_count_steps': None, '_evaluation_master': '', '_global_id_in_cluster': 0, '_master': ''}
INFO:tensorflow:_TPUContext: eval_on_tpu True
WARNING:tensorflow:eval_on_tpu ignored because use_tpu is False.
INFO:tensorflow:Writing example 0 of 8802
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: dev-1
INFO:tensorflow:tokens: [CLS] 开 初 婚 未 育 证 明 怎 么 弄 ？ [SEP] 初 婚 未 育 情 况 证 明 怎 么 开 ？ [SEP]
INFO:tensorflow:input_ids: 101 2458 1159 2042 3313 5509 6395 3209 2582 720 2462 8043 102 1159 2042 3313 5509 2658 1105 6395 3209 2582 720 2458 8043 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 1 (id = 1)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: dev-2
INFO:tensorflow:tokens: [CLS] 谁 知 道 她 是 网 络 美 女 吗 ？ [SEP] 爱 情 这 杯 酒 谁 喝 都 会 醉 是 什 么 歌 [SEP]
INFO:tensorflow:input_ids: 101 6443 4761 6887 1961 3221 5381 5317 5401 1957 1408 8043 102 4263 2658 6821 3344 6983 6443 1600 6963 833 7004 3221 784 720 3625 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: dev-3
INFO:tensorflow:tokens: [CLS] 人 和 畜 生 的 区 别 是 什 么 ？ [SEP] 人 与 畜 生 的 区 别 是 什 么 ！ [SEP]
INFO:tensorflow:input_ids: 101 782 1469 4523 4495 4638 1277 1166 3221 784 720 8043 102 782 680 4523 4495 4638 1277 1166 3221 784 720 8013 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 1 (id = 1)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: dev-4
INFO:tensorflow:tokens: [CLS] 男 孩 喝 女 孩 的 尿 的 故 事 [SEP] 怎 样 才 知 道 是 生 男 孩 还 是 女 孩 [SEP]
INFO:tensorflow:input_ids: 101 4511 2111 1600 1957 2111 4638 2228 4638 3125 752 102 2582 3416 2798 4761 6887 3221 4495 4511 2111 6820 3221 1957 2111 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: dev-5
INFO:tensorflow:tokens: [CLS] 这 种 图 片 是 用 什 么 软 件 制 作 的 ？ [SEP] 这 种 图 片 制 作 是 用 什 么 软 件 呢 ？ [SEP]
INFO:tensorflow:input_ids: 101 6821 4905 1745 4275 3221 4500 784 720 6763 816 1169 868 4638 8043 102 6821 4905 1745 4275 1169 868 3221 4500 784 720 6763 816 1450 8043 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 1 (id = 1)
INFO:tensorflow:***** Running evaluation *****
INFO:tensorflow:  Num examples = 8802 (8802 actual, 0 padding)
INFO:tensorflow:  Batch size = 8
INFO:tensorflow:Could not find trained model in model_dir: ./lcqmc_mt_cut_l6h384i1200i/, running initialization to evaluate.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Running eval on CPU
INFO:tensorflow:*** Features ***
INFO:tensorflow:  name = input_ids, shape = (?, 128)
INFO:tensorflow:  name = input_mask, shape = (?, 128)
INFO:tensorflow:  name = is_real_example, shape = (?,)
INFO:tensorflow:  name = label_ids, shape = (?,)
INFO:tensorflow:  name = segment_ids, shape = (?, 128)
bert/embeddings/word_embeddings ppp <tf.Variable 'bert/embeddings/word_embeddings:0' shape=(21128, 384) dtype=float32_ref>
bert/embeddings/token_type_embeddings ppp <tf.Variable 'bert/embeddings/token_type_embeddings:0' shape=(2, 384) dtype=float32_ref>
bert/embeddings/position_embeddings ppp <tf.Variable 'bert/embeddings/position_embeddings:0' shape=(512, 384) dtype=float32_ref>
bert/embeddings/LayerNorm/beta ppp <tf.Variable 'bert/embeddings/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/embeddings/LayerNorm/gamma ppp <tf.Variable 'bert/embeddings/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_0/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_0/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_0/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_0/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_0/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_0/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_0/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_1/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_1/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_1/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_1/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_1/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_1/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_1/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_2/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_2/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_2/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_2/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_2/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_2/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_2/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>2020-07-15 15:03:05.918291: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX512F
2020-07-15 15:03:06.354984: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:1d:00.0
totalMemory: 31.72GiB freeMemory: 29.68GiB
2020-07-15 15:03:06.355241: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2020-07-15 15:03:06.853748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-15 15:03:06.854093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2020-07-15 15:03:06.854276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2020-07-15 15:03:06.854856: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 28793 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:1d:00.0, compute capability: 7.0)
INFO:tensorflow:**** Trainable Variables ****
INFO:tensorflow:  name = bert/embeddings/word_embeddings:0, shape = (21128, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/token_type_embeddings:0, shape = (2, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/position_embeddings:0, shape = (512, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/pooler/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/pooler/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = output_weights:0, shape = (2, 384)
INFO:tensorflow:  name = output_bias:0, shape = (2,)
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2020-07-15-07:03:07
INFO:tensorflow:Graph was finalized.
2020-07-15 15:03:08.046619: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2020-07-15 15:03:08.046862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-15 15:03:08.047181: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2020-07-15 15:03:08.047305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2020-07-15 15:03:08.047673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 28793 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:1d:00.0, compute capability: 7.0)
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2020-07-15-07:03:18
INFO:tensorflow:Saving dict for global step 0: eval_accuracy = 0.4998864, eval_loss = 0.69319314, global_step = 0, loss = 0.6932033
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.4998864
INFO:tensorflow:  eval_loss = 0.69319314
INFO:tensorflow:  global_step = 0
INFO:tensorflow:  loss = 0.6932033
INFO:tensorflow:Writing example 0 of 12500
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-1
INFO:tensorflow:tokens: [CLS] 谁 有 狂 三 这 张 高 清 的 [SEP] 这 张 高 清 图 ， 谁 有 [SEP]
INFO:tensorflow:input_ids: 101 6443 3300 4312 676 6821 2476 7770 3926 4638 102 6821 2476 7770 3926 1745 8024 6443 3300 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-2
INFO:tensorflow:tokens: [CLS] 英 雄 联 盟 什 么 英 雄 最 好 [SEP] 英 雄 联 盟 最 好 英 雄 是 什 么 [SEP]
INFO:tensorflow:input_ids: 101 5739 7413 5468 4673 784 720 5739 7413 3297 1962 102 5739 7413 5468 4673 3297 1962 5739 7413 3221 784 720 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 1 (id = 1)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-3
INFO:tensorflow:tokens: [CLS] 这 是 什 么 意 思 ， 被 蹭 网 吗 [SEP] 我 也 是 醉 了 ， 这 是 什 么 意 思 [SEP]
INFO:tensorflow:input_ids: 101 6821 3221 784 720 2692 2590 8024 6158 6701 5381 1408 102 2769 738 3221 7004 749 8024 6821 3221 784 720 2692 2590 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-4
INFO:tensorflow:tokens: [CLS] 现 在 有 什 么 动 画 片 好 看 呢 ？ [SEP] 现 在 有 什 么 好 看 的 动 画 片 吗 ？ [SEP]
INFO:tensorflow:input_ids: 101 4385 1762 3300 784 720 1220 4514 4275 1962 4692 1450 8043 102 4385 1762 3300 784 720 1962 4692 4638 1220 4514 4275 1408 8043 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 1 (id = 1)
INFO:tensorflow:*** Example ***
INFO:tensorflow:guid: test-5
INFO:tensorflow:tokens: [CLS] 请 问 晶 达 电 子 厂 现 在 的 工 资 待 遇 怎 么 样 要 求 有 哪 些 [SEP] 三 星 电 子 厂 工 资 待 遇 怎 么 样 啊 [SEP]
INFO:tensorflow:input_ids: 101 6435 7309 3253 6809 4510 2094 1322 4385 1762 4638 2339 6598 2521 6878 2582 720 3416 6206 3724 3300 1525 763 102 676 3215 4510 2094 1322 2339 6598 2521 6878 2582 720 3416 1557 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:segment_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
INFO:tensorflow:label: 0 (id = 0)
INFO:tensorflow:Writing example 10000 of 12500
INFO:tensorflow:***** Running prediction*****
INFO:tensorflow:  Num examples = 12500 (12500 actual, 0 padding)
INFO:tensorflow:  Batch size = 8
INFO:tensorflow:Could not find trained model in model_dir: ./lcqmc_mt_cut_l6h384i1200i/, running initialization to evaluate.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Running eval on CPU
INFO:tensorflow:*** Features ***
INFO:tensorflow:  name = input_ids, shape = (?, 128)
INFO:tensorflow:  name = input_mask, shape = (?, 128)
INFO:tensorflow:  name = is_real_example, shape = (?,)
INFO:tensorflow:  name = label_ids, shape = (?,)
INFO:tensorflow:  name = segment_ids, shape = (?, 128)

bert/encoder/layer_3/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_3/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_3/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_3/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_3/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_3/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_3/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_3/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_4/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_4/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_4/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_4/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_4/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_4/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_4/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_5/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_5/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_5/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_5/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_5/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_5/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_5/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/pooler/dense/kernel ppp <tf.Variable 'bert/pooler/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/pooler/dense/bias ppp <tf.Variable 'bert/pooler/dense/bias:0' shape=(384,) dtype=float32_ref>
output_weights ppp <tf.Variable 'output_weights:0' shape=(2, 384) dtype=float32_ref>
output_bias ppp <tf.Variable 'output_bias:0' shape=(2,) dtype=float32_ref>
bert/embeddings/word_embeddings ppp <tf.Variable 'bert/embeddings/word_embeddings:0' shape=(21128, 384) dtype=float32_ref>
bert/embeddings/token_type_embeddings ppp <tf.Variable 'bert/embeddings/token_type_embeddings:0' shape=(2, 384) dtype=float32_ref>
bert/embeddings/position_embeddings ppp <tf.Variable 'bert/embeddings/position_embeddings:0' shape=(512, 384) dtype=float32_ref>
bert/embeddings/LayerNorm/beta ppp <tf.Variable 'bert/embeddings/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/embeddings/LayerNorm/gamma ppp <tf.Variable 'bert/embeddings/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_0/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_0/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_0/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_0/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_0/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_0/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_0/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_1/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_1/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_1/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_1/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_1/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_1/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_1/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_2/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_2/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_2/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_2/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_2/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_2/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_2/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_3/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_3/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder2020-07-15 15:03:30.348827: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2020-07-15 15:03:30.348978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-15 15:03:30.349098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2020-07-15 15:03:30.349212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2020-07-15 15:03:30.349604: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 28793 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:1d:00.0, compute capability: 7.0)
INFO:tensorflow:**** Trainable Variables ****
INFO:tensorflow:  name = bert/embeddings/word_embeddings:0, shape = (21128, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/token_type_embeddings:0, shape = (2, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/position_embeddings:0, shape = (512, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/pooler/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/pooler/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = output_weights:0, shape = (2, 384)
INFO:tensorflow:  name = output_bias:0, shape = (2,)
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2020-07-15-07:03:30
INFO:tensorflow:Graph was finalized.
2020-07-15 15:03:31.041540: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2020-07-15 15:03:31.041802: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-15 15:03:31.041999: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2020-07-15 15:03:31.042176: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2020-07-15 15:03:31.042550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 28793 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:1d:00.0, compute capability: 7.0)
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2020-07-15-07:03:44
INFO:tensorflow:Saving dict for global step 0: eval_accuracy = 0.5, eval_loss = 0.69386685, global_step = 0, loss = 0.6938668
INFO:tensorflow:***** test results *****
INFO:tensorflow:  eval_accuracy = 0.5
INFO:tensorflow:  eval_loss = 0.69386685
INFO:tensorflow:  global_step = 0
INFO:tensorflow:  loss = 0.6938668
INFO:tensorflow:***** Predict results *****
INFO:tensorflow:Could not find trained model in model_dir: ./lcqmc_mt_cut_l6h384i1200i/, running initialization to predict.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Running infer on CPU
INFO:tensorflow:*** Features ***
INFO:tensorflow:  name = input_ids, shape = (?, 128)
INFO:tensorflow:  name = input_mask, shape = (?, 128)
INFO:tensorflow:  name = is_real_example, shape = (?,)
INFO:tensorflow:  name = label_ids, shape = (?,)
INFO:tensorflow:  name = segment_ids, shape = (?, 128)
/layer_3/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_3/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_3/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_3/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_3/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_3/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_4/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_4/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_4/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_4/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_4/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_4/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_4/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_5/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_5/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_5/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_5/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_5/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_5/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_5/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/pooler/dense/kernel ppp <tf.Variable 'bert/pooler/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/pooler/dense/bias ppp <tf.Variable 'bert/pooler/dense/bias:0' shape=(384,) dtype=float32_ref>
output_weights ppp <tf.Variable 'output_weights:0' shape=(2, 384) dtype=float32_ref>
output_bias ppp <tf.Variable 'output_bias:0' shape=(2,) dtype=float32_ref>
bert/embeddings/word_embeddings ppp <tf.Variable 'bert/embeddings/word_embeddings:0' shape=(21128, 384) dtype=float32_ref>
bert/embeddings/token_type_embeddings ppp <tf.Variable 'bert/embeddings/token_type_embeddings:0' shape=(2, 384) dtype=float32_ref>
bert/embeddings/position_embeddings ppp <tf.Variable 'bert/embeddings/position_embeddings:0' shape=(512, 384) dtype=float32_ref>
bert/embeddings/LayerNorm/beta ppp <tf.Variable 'bert/embeddings/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/embeddings/LayerNorm/gamma ppp <tf.Variable 'bert/embeddings/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_0/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_0/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_0/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_0/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_0/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_0/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_0/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_0/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_0/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_0/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_0/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_1/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_1/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_1/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_1/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_1/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_1/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_1/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_1/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_1/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_1/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_1/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_2/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_2/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_2/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_2/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_2/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_2/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_2/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_2/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_2/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_2/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_2/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_3/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_3/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_3/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_3/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_3/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_3/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_3/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_3/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_3/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_3/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_3/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/lay2020-07-15 15:03:49.481672: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2020-07-15 15:03:49.481986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-15 15:03:49.482115: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2020-07-15 15:03:49.482230: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2020-07-15 15:03:49.482590: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 28793 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:1d:00.0, compute capability: 7.0)
INFO:tensorflow:**** Trainable Variables ****
INFO:tensorflow:  name = bert/embeddings/word_embeddings:0, shape = (21128, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/token_type_embeddings:0, shape = (2, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/position_embeddings:0, shape = (512, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/embeddings/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_0/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_1/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_2/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_3/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_4/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/query/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/key/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/self/value/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/attention/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/kernel:0, shape = (384, 1200), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/intermediate/dense/bias:0, shape = (1200,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/kernel:0, shape = (1200, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/beta:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/encoder/layer_5/output/LayerNorm/gamma:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/pooler/dense/kernel:0, shape = (384, 384), *INIT_FROM_CKPT*
INFO:tensorflow:  name = bert/pooler/dense/bias:0, shape = (384,), *INIT_FROM_CKPT*
INFO:tensorflow:  name = output_weights:0, shape = (2, 384)
INFO:tensorflow:  name = output_bias:0, shape = (2,)
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Graph was finalized.
2020-07-15 15:03:50.051476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2020-07-15 15:03:50.051703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-07-15 15:03:50.051863: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2020-07-15 15:03:50.052055: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2020-07-15 15:03:50.052474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 28793 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:1d:00.0, compute capability: 7.0)
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
er_4/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_4/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_4/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_4/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_4/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_4/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_4/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_4/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_4/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_4/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_4/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_4/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/query/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/query/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/query/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/query/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/key/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/key/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/key/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/key/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/self/value/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/self/value/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/self/value/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/self/value/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/attention/output/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/encoder/layer_5/attention/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/attention/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_5/attention/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/attention/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_5/attention/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/intermediate/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/intermediate/dense/kernel:0' shape=(384, 1200) dtype=float32_ref>
bert/encoder/layer_5/intermediate/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/intermediate/dense/bias:0' shape=(1200,) dtype=float32_ref>
bert/encoder/layer_5/output/dense/kernel ppp <tf.Variable 'bert/encoder/layer_5/output/dense/kernel:0' shape=(1200, 384) dtype=float32_ref>
bert/encoder/layer_5/output/dense/bias ppp <tf.Variable 'bert/encoder/layer_5/output/dense/bias:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/output/LayerNorm/beta ppp <tf.Variable 'bert/encoder/layer_5/output/LayerNorm/beta:0' shape=(384,) dtype=float32_ref>
bert/encoder/layer_5/output/LayerNorm/gamma ppp <tf.Variable 'bert/encoder/layer_5/output/LayerNorm/gamma:0' shape=(384,) dtype=float32_ref>
bert/pooler/dense/kernel ppp <tf.Variable 'bert/pooler/dense/kernel:0' shape=(384, 384) dtype=float32_ref>
bert/pooler/dense/bias ppp <tf.Variable 'bert/pooler/dense/bias:0' shape=(384,) dtype=float32_ref>
output_weights ppp <tf.Variable 'output_weights:0' shape=(2, 384) dtype=float32_ref>
output_bias ppp <tf.Variable 'output_bias:0' shape=(2,) dtype=float32_ref>
