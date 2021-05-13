export BERT_BASE_DIR=/home/hadoop-aipnlp/cephfs/data/bujiahao/bert-base/
export BERT_model=/home/hadoop-aipnlp/cephfs/data/BERT_TRAINING_SERVICE/platform/work_space/aipnlp/models/1/model.ckpt-692000
#export BERT_model=/home/hadoop-aipnlp/cephfs/data/bujiahao/output/coling-output/s-bert-ner-attention-0.2/model.ckpt-17534
export OUTPUT_DIR=/home/hadoop-aipnlp/cephfs/data/bujiahao/output/coling-output/s-bert-ner-sm-KLattention-0.2
export CUDA_VISIBLE_DEVICES=1
python s_bert_ner_sm_attention.py \
 --task_name=meituan \
 --do_train=true \
 --do_eval=true \
 --do_predict=true \
 --do_save=false \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$BERT_model \
 --max_seq_length=32 \
 --train_batch_size=64 \
 --eval_batch_size=2048 \
 --predict_batch_size=2048 \
 --learning_rate=2e-5 \
 --num_train_epochs=2 \
 --pooling_strategy=mean \
 --scale_rate=0.2 \
 --train_data_path=/home/hadoop-aipnlp/cephfs/data/bujiahao/data/coling/poi_train_ner/poi_train.csv \
 --eval_data_path=/home/hadoop-aipnlp/cephfs/data/bujiahao/data/coling/poi_test_ner/poi_test.csv \
 --predict_data_path=/home/hadoop-aipnlp/cephfs/data/bujiahao/data/coling/poi_test_ner/poi_test.csv \
 --ner_label_file=/home/hadoop-aipnlp/cephfs/data/bujiahao/data/ner.63k/labels.txt \
 --output_dir=$OUTPUT_DIR/

