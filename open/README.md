#### Environment
Tested on Ubuntu 16.04, Tesla V100 (32GB), Python3.7

#### Requirements
- tensorflow==1.15.0
- ujson==1.35
- tokenizers==0.8.0
- json_lines==0.5.0
- numpy==1.16.5

#### Datasets & Params
downloading datasets to **{VIRT_DIR}/datasets**
- GLUE: https://gluebenchmark.com/
- SuperGLUE: https://super.gluebenchmark.com

the dataset dir should look like below (use tree -L 2 ${VIRT_DIR}/datasets):
```
${VIRT_dir}$/datasets
├── boolq
│   ├── test.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── mnli
│   ├── dev_mismatched.tsv
│   └── train.tsv
├── qqp
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
|── rte
│   ├── dev.json
│   ├── test.json
│   └── train.json
```
split 10% of train for tuning hyper-parameters:
```
cat boolq-train.jsonl | shuf > boolq-train-shuf.jsonl
head -n943 boolq-train-shuf.jsonl > boolq-tune.jsonl
tail -n8484 boolq-train-shuf.jsonl > boolq-train.jsonl

cat rte-train.jsonl | shuf > rte-train-shuf.jsonl
head -n294 race-train-shuf.jsonl > rte-tune.jsonl
tail -n2646 race-train-shuf.jsonl > rte-train.jsonl

cat qqp-train.jsonl | shuf > qqp-train-shuf.jsonl
head -n36385 qqp-train-shuf.jsonl > qqp-tune.jsonl
tail -n327464 qqp-train-shuf.jsonl > qqp-train.jsonl

cat mnli-train.jsonl | shuf > mnli-train-shuf.jsonl
head -n39270 mnli-train-shuf.jsonl > mnli-tune.jsonl
tail -n353432 mnli-train-shuf.jsonl > mnli-train.jsonl
```

download bert.vocab at this [link](https://github.com/StonyBrookNLP/deformer/releases/download/v1.0/bert.vocab)

download bert pre-trained models at this [link](https://github.com/google-research/bert)

#### Training and Evaluation

##### Interaction-based models
`python bert_mnli_qqp.py` or `python bert_rte_boolq.py` specify task by `--task`, eval is similar. see below example commands for mnli:
```shell script
export BERT_BASE_DIR={YOUR BERT_BASE_DIR}
export MY_DATASET={YOUR DATASET_DIR}
export OUTPUT={YOUR OUTPUT_DIR}
python bert_mnli_qqp.py \
  --task_name=mnli \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=259 \
  --train_batch_size=5 \
  --learning_rate=5e-5 \
  --num_train_epochs=6.0 \
  --output_dir=$OUTPUT \
  --pooling_strategy=mean
```

##### Representation-based models
`python representation-based_mnli_qqp.py` or `python representation-based_rte_boolq.py` specify task by `--task`, model by `--model_type`. eval is similar. If you want to enbale virt, please specify `--use_virt=true`. see below example commands for mnli for our virt-encoder:
```shell script
export BERT_MODEL_DIR={YOUR BERT_BASE_DIR}
export TEACHER_MODEL_DIR={FINETUNED BERT MODEL DIR}
export MY_DATASET={YOUR DATASET DIR}
export OUTPUT={YOUR OUTPUT DIR}
python3 representation-based_mnli_qqp.py \
  --task_name=mnli \
  --pooling_strategy=mean \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint_teacher=$TEACHER_MODEL_DIR/model.ckpt-{numbers} \
  --init_checkpoint_student=$BERT_MODEL_DIR/bert_model.ckpt \
  --max_seq_length_bert=259 \
  --max_seq_length_sbert=130 \
  --train_batch_size=28 \
  --learning_rate=5e-5 \
  --num_train_epochs=5 \
  --do_save=false \
  --output_dir=$OUTPUT \
  --use_virt=true \
  --kd_weight_att=0.4 \
  --model_type=virta
```

#### Evaluate FLOPs
To evaluate FLOPs, use `flops*.py`. see below example commands for boolq in bert_base:

```shell script
export BERT_BASE_DIR=${BERT_BASE_DIR}$
export MY_DATASET={YOUR DATASET DIR}
export OUTPUT={YOUR OUTPUT DIR}
python flops_bert.py \
  --task_name=boolq \
  --do_train=false \
  --do_eval=true \
  --do_predict=false \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=392 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$OUTPUT \
  --pooling_strategy=mean
```