# Network Traffic Model


## Outline

- 1. Requirements
- 2. Reproducing the results in the paper
    - 2.1. Finetuning for Traffic Understanding Tasks
    - 2.2. Finetuning for Traffic Generation Tasks
- 3. Data Preprocessing
    - 3.1. Data Preprocessing for Model Pre-training
        - 3.1.1. Converting traffic into a corpus
        - 3.1.2. Processing the pre-trained corpus
    - 3.2. Data Preprocessing for Traffic Understanding Tasks
    - 3.3. Data Preprocessing for Traffic Generation Tasks
- 4. Model Pre-training
- 5. Citation


## 1. Requirements
```
Python >= 3.6
CUDA: 11.4
torch >= 1.1
six >= 1.12.0
scapy == 2.4.4
numpy == 1.19.2
shutil, random, json, pickle, binascii, flowcontainer, argparse, packaging, tshark
```

## 2. Reproducing the results in the paper

You can fine-tune the model using [pre-trained model](https://drive.google.com/file/d/1GNbWtVgrG9XcuApgkSl1hDRTsDXqm12R/view?usp=drive_link) and processed dataset we provide (we will give the corresponding files below). In addition, we will give the details of data preprocessing and model pretraining later.


### 2.1. Finetuning for Traffic Understanding Tasks

**With Qwen:**
```
python3 finetune/run_understanding.py  --pretrained_model_path pretrained_model.bin \
                                                  --output_model_path models/finetuned_model.bin \
                                                  --vocab_path models/encryptd_vocab.txt \
                                                  --config_path models/qwen/config.json \
                                                  --use_qwen \
                                                  --qwen_model_name Qwen/Qwen2.5-0.5B \
                                                  --train_path finetune_dataset/train_dataset.tsv \
                                                  --dev_path finetune_dataset/valid_dataset.tsv \
                                                  --test_path finetune_dataset/test_dataset.tsv \
                                                  --epochs_num 10 \
                                                  --batch_size 32 \
                                                  --labels_num 2 \
                                                  --pooling mean
```

### 2.2. Finetuning for Traffic Generation Tasks

**With Qwen:**
```
python3 finetune/run_generation.py    --pretrained_model_path pretrained_model.bin \
                                      --output_model_path models/finetuned_model.bin \
                                      --vocab_path models/encryptd_vocab.txt \
                                      --config_path models/qwen/config.json \
                                      --use_qwen \
                                      --qwen_model_name Qwen/Qwen2.5-0.5B \
                                      --train_path datasets/train_dataset.tsv \
                                      --dev_path datasets/valid_dataset.tsv \
                                      --test_path datasets/test_dataset.tsv \
                                      --learning_rate 1e-5 \
                                      --epochs_num 10 \
                                      --batch_size 16 \
                                      --pooling mean \
                                      --seq_length 256 \
                                      --tgt_seq_length 4
``

## 3. Data Preprocessing

### 3.1. Data Preprocessing for Model Pre-training

#### 3.1.1. Converting traffic into a corpus

In order to pre-train the model, we first need to convert the traffic data into a corpus. Note you'll need to change the file paths and some configures at the top of the "main.py" file. Specifically, you need to

1. set the variable pcap_path as the directory of PCAP data to be processed.
2. set the variable word_dir and word_name as the storage directory of pre-training daraset.
3. set the variable output_split_path and pcap_output_path. The pcap_output_path indicates the storage directory where the pcapng format of PCAP data is converted to pcap format. The output_split_path represents the storage directory for PCAP data slicing into session format.

Finally, you can gnerate pre-training corpus by following the completion of PCAP data processing.

```
python3 pre-process/main.py
```

#### 3.1.2. Processing the pre-trained corpus

```
python3 preprocess.py   --corpus_path corpora/traffic.txt \
                        --vocab_path models/encryptd_vocab.txt \
                        --dataset_path distributed/dataset.pt \
                        --processes_num 8 \
                        --data_processor lm

```

### 3.2. Data Preprocessing for Traffic Understanding Tasks

```
python3 pre-process/input_generation_understanding.py   --pcap_path "data/pcap/" \
                                                        --dataset_dir "data/understanding/datasets/" \
                                                        --class_num 17 \
                                                        --middle_save_path "data/understanding/result/" \
                                                        --random_seed 01
```

### 3.3. Data Preprocessing for Traffic Generation Tasks


```
python3 pre-process/input_generation_generation.py  --pcap_path "data/pcap/" \
                                                    --dataset_dir "data/generation/datasets/" \
                                                    --class_num 17 \
                                                    --middle_save_path "data/generation/result/" \
                                                    --random_seed 01
```


## 4. Model Pre-training


### 4.1. Pre-training with Transformer (Original)

```
python3 pretrain.py   --dataset_path distributed/dataset.pt \
                      --vocab_path models/encryptd_vocab.txt \
                      --config_path models/transformer/config.json \
                      --output_model_path pretrained_model.bin \
                      --world_size 8 \
                      --gpu_ranks 0 1 2 3 4 5 6 7 \
                      --learning_rate 1e-4 \
                      --data_processor lm \
                      --embedding word pos \
                      --remove_embedding_layernorm \
                      --encoder transformer \
                      --mask causal \
                      --layernorm_positioning pre \
                      --target lm \
                      --tie_weights
```

### 4.2. Pre-training with Qwen/Qwen2.5-0.5B

**With GPU:**
```
python3 pretrain.py   --dataset_path distributed/dataset.pt \
                      --vocab_path models/encryptd_vocab.txt \
                      --config_path models/qwen/config.json \
                      --output_model_path pretrained_model.bin \
                      --use_qwen \
                      --qwen_model_name Qwen/Qwen2.5-0.5B \
                      --world_size 1 \
                      --gpu_ranks 0 \
                      --learning_rate 1e-4 \
                      --data_processor lm \
                      --batch_size 32 \
                      --total_steps 100000
```

**Without GPU (CPU only):**
```
python3 pretrain.py   --dataset_path distributed/dataset.pt \
                      --vocab_path models/encryptd_vocab.txt \
                      --config_path models/qwen/config.json \
                      --output_model_path pretrained_model.bin \
                      --use_qwen \
                      --qwen_model_name Qwen/Qwen2.5-0.5B \
                      --world_size 1 \
                      --gpu_ranks \
                      --learning_rate 1e-4 \
                      --data_processor lm \
                      --batch_size 8 \
                      --total_steps 100000
```

