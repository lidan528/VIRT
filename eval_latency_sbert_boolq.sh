export CUDA_VISIBLE_DEVICES=0
#export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/bert_base
#export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/bujiahao/outstanding_ckpt/128_model
#export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/bert_base/train_log/bert_distil_minilm_L6h128I512A2_common128_wa
export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/mnli/ #全局变量 数据集所在地址
#export OUTPUT=./output/search_spuall_3/
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/mnli/s_bert_base_pad_sep_realpool/
python eval_flops_sbert.py \
  --task_name=boolq \
  --pooling_strategy=mean \
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --max_seq_length_query=64 \
  --max_seq_length_doc=328 \
  --train_batch_size=28 \
  --learning_rate=5e-5 \
  --test_flie_name=None \
  --num_train_epochs=5.0 \
  --do_save=false \
  --output_dir=$OUTPUT
