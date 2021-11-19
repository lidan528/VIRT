export CUDA_VISIBLE_DEVICES=1
#export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/bert_base
#export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/bujiahao/outstanding_ckpt/128_model
#export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/bert_base/train_log/bert_distil_minilm_L6h128I512A2_common128_wa
export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export TEACHER_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/rte/bert_base_bipartition_mean_pool
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/rte/ #全局变量 数据集所在地址
#export OUTPUT=./output/search_spuall_3/
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/rte/bi-encoder_wo_att/
python _--sbert_distill_boolq.py \
  --task_name=rte \
  --pooling_strategy=mean \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint_teacher=$TEACHER_MODEL_DIR/model.ckpt-2000 \
  --init_checkpoint_student=$BERT_MODEL_DIR/bert_model.ckpt \
  --max_seq_length_query=64 \
  --max_seq_length_doc=328 \
  --train_batch_size=28 \
  --learning_rate=5e-5 \
  --test_flie_name=None \
  --num_train_epochs=30 \
  --do_save=false \
  --output_dir=$OUTPUT \
  --use_kd_logit_kl=false \
  --use_kd_logit_mse=false \
  --kd_weight_logit=0.4 \
  --use_kd_att=false \
  --kd_weight_att=700 \
  --use_resnet_predict=false \
  --model_type=bi_encoder