export CUDA_VISIBLE_DEVICES=0
export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/squad/ #全局变量 数据集所在地址
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/squad/bert_base_bipartition/
python _run_classifier.py \
  --task_name=squad \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=40 \
  --learning_rate=5e-5 \
  --num_train_epochs=6.0 \
  --output_dir=$OUTPUT \
