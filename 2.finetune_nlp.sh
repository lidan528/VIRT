export CUDA_VISIBLE_DEVICES=6
export BERT_BASE_DIR=/opt/meituan/cephfs/user/hadoop-aipnlp/jingang.wjg/NER/nlp_bert/wiki_baidu-char_01_bak12_mark
export MY_DATASET=/opt/meituan/cephfs/user/hadoop-aipnlp/jingang.wjg/LCQMC/train_data/ #全局变量 数据集所在地址
export OUTPUT=/opt/meituan/cephfs/user/hadoop-aipnlp/jingang.wjg/LCQMC/lcqmc_nlp_output/
python run_classifier.py \
  --task_name=lcqmc \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT \
