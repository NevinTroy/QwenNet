# QwenNet  
### Transformer-Based Traffic Understanding and Generation Using Qwen 2.5 0.5B

QwenNet is a transformer-based framework for encrypted network traffic
classification and next-token traffic generation. The system processes raw PCAP
files, converts them into bigram token sequences, and fine-tunes the
Qwen2.5-0.5B encoder on flow-level representations.

This repository includes:
- End-to-end preprocessing pipeline  
- Scripts for dataset preparation  
- Fine-tuning for **traffic understanding**  
- Fine-tuning for **traffic generation**  
- Example notebook and training utilities  

All components have been tested on Google Colab and a local Ubuntu environment.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
Or manually:

```bash
pip install torch scapy tqdm pandas numpy scikit-learn tokenizers psutil scipy
```
## Dataset Downloads

This project uses two publicly available datasets:

| Dataset         | Usage                                   | Link |
|-----------------|-------------------------------------------|------|
| **USTC-TFC2016** | Application Traffic Classification        | [Download](https://github.com/davidyslu/USTC-TFC2016) |
| **ISCX-VPN-2016** | VPN / Non-VPN Traffic Classification      | [Download](https://www.unb.ca/cic/datasets/vpn.html) |

Download the cleaned data from: [Drive Link](https://drive.google.com/drive/folders/1ePQxWhdiKSdiq9rhdXJN8_lcALrnCcX_?usp=sharing)

## Data Preparation

### Step 1 — Preprocess PCAPs into Training TSV Files

This script performs the following operations:

- Extracts up to **32 packets per flow**  
- **Anonymizes metadata** for privacy and consistency  
- Converts **payload bytes into hex format**  
- Segments the hex sequence into **bigrams**  
- Outputs **flow-level sequences** as TSV files suitable for QwenNet preprocessing and model training  

```bash
python3 pre-process/main.py
```

### Step 2 - Convert to pretraining corpus
```bash
python3 preprocess.py   --corpus_path corpora/traffic.txt \
                        --vocab_path models/encryptd_vocab.txt \
                        --dataset_path distributed/dataset.pt \
                        --processes_num 8 \
                        --data_processor lm
```

## Pretraining

```bash
!python3 pretrain.py   --dataset_path distributed/dataset.pt \
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
                      --total_steps 10000
```

## Fine Tuning for Traffic Understanding

Run:
```bash
python3 finetune/run_understanding.py \
    --pretrained_model_path models/pretrained_model.bin \
    --output_model_path models/finetuned_model.bin \
    --vocab_path models/encryptd_vocab.txt \
    --config_path models/qwen/config.json \
    --use_qwen \
    --qwen_model_name Qwen/Qwen2.5-0.5B \
    --train_path data/understanding_dataset/train_dataset.tsv \
    --dev_path data/understanding_dataset/valid_dataset.tsv \
    --test_path data/understanding_datasets/test_dataset.tsv \
    --epochs_num 2 \
    --batch_size 32 \
    --labels_num N \
    --pooling mean
```
## Fine Tuning for Traffic Generation

Run:
```bash
python3 finetune/run_generation.py \
    --pretrained_model_path models/pretrained_model.bin \
    --output_model_path models/finetuned_model_gen.bin \
    --vocab_path models/encryptd_vocab.txt \
    --config_path models/qwen/config.json \
    --use_qwen \
    --qwen_model_name Qwen/Qwen2.5-0.5B \
    --train_path data/generation_dataset/train_dataset.tsv \
    --dev_path data/generation_dataset/valid_dataset.tsv \
    --test_path data/generation_dataset/test_dataset.tsv \
    --learning_rate 1e-5 \
    --epochs_num 2 \
    --batch_size 16 \
    --seq_length 256 \
    --tgt_seq_length 4 \
    --pooling mean
```
