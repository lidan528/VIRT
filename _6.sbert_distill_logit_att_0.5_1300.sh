export CUDA_VISIBLE_DEVICES=2
export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export TEACHERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/squad/bert_base_bipartition
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/squad #全局变量 数据集所在地址
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/squad/sbert_distill_logit_att_0.5_1300/
python _sbert_distill.py \
  --task_name=squad \
  --train_file=$MY_DATASET/squad_v1.1-train.jsonl \
  --dev_file=$MY_DATASET/squad_v1.1-dev.jsonl \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --init_checkpoint_teacher=$TEACHERT_BASE_DIR/model.ckpt-13000 \
  --train_batch_size=28 \
  --learning_rate=5e-5 \
  --pooling_strategy=mean \
  --num_train_epochs=6.0 \
  --output_dir=$OUTPUT \
  --use_kd_logit=true \
  --kd_weight_logit=0.5 \
  --use_kd_att=true \
  --kd_weight_att=1300
