export CUDA_VISIBLE_DEVICES=0
export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/checkpoints/roberta_wwm_large_ext #全局变量 下载的预训练BERT地址
export MY_DATASET=./train_data/ #全局变量 数据集所在地址
export OUTPUT=./lcqmc_output/robertalarge1/
python run_classifier.py \
  --task_name=lcqmc \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT \
