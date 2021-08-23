export CUDA_VISIBLE_DEVICES=7
#export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/bert_base
#export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/bujiahao/outstanding_ckpt/128_model
#export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/transfer_to_zw03/yangyang113/bert_base/train_log/bert_distil_minilm_L6h128I512A2_common128_wa
export BERT_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/english_bert_base_model/uncased_L-12_H-768_A-12
export TEACHER_MODEL_DIR=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/mnli/bert_base_bipartition_mean_pool
export MY_DATASET=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/dataset/mnli/ #全局变量 数据集所在地址
#export OUTPUT=./output/search_spuall_3/
export OUTPUT=/home/hadoop-aipnlp/cephfs/data/lidan65/distill/output/mnli/dipair_logit/
python3 sbert_distill.py \
  --task_name=mnli \
  --pooling_strategy=mean \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint_teacher=$TEACHER_MODEL_DIR/model.ckpt-43000 \
  --init_checkpoint_student=$BERT_MODEL_DIR/bert_model.ckpt \
  --max_seq_length_bert=259 \
  --max_seq_length_sbert=130 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --test_flie_name=None \
  --num_train_epochs=6 \
  --do_save=false \
  --output_dir=$OUTPUT \
  --use_kd_logit_kl=true \
  --use_kd_logit_mse=false \
  --kd_weight_logit=0.4 \
  --use_kd_att=false \
  --kd_weight_att=5 \
  --use_all_layer_emb=false \
  --use_resnet_predict=false \
  --use_weighted_att=false \
  --use_att_head=false \
  --model_type=dipair