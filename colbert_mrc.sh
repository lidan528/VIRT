export CUDA_VISIBLE_DEVICES=3
export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/squad #全局变量 数据集所在地址
colbert_dim=128
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/squad/colbert_${colbert_dim}_meanpool/
python colbert_mrc.py \
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
  --train_batch_size=32 \
  --colbert_dim=$colbert_dim \
  --learning_rate=5e-5 \
  --num_train_epochs=6.0 \
  --output_dir=$OUTPUT \
  --pooling_strategy=mean
