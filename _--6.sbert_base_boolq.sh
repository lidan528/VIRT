export CUDA_VISIBLE_DEVICES=4
export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/boolq/ #全局变量 数据集所在地址
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/boolq/sbert_base_bipartition/
python _--sbert_boolq.py \
  --task_name=boolq \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length_query=64 \
  --max_seq_length_doc=328 \
  --train_batch_size=28 \
  --learning_rate=5e-5 \
  --num_train_epochs=30.0 \
  --output_dir=$OUTPUT \
  --pooling_strategy=mean