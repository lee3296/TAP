#!/bin/bash

EXP_NAME="main_train" #main_train #"ablation_no_distill" #twotimes
SEED=41 #41-44         
DEVICE="cuda"
GPU=0

# Federated learning parameters
NUM_ROUNDS=200 
CLIENTS_PER_ROUND=2 
LOCAL_ROUNDS=20  # FLAVA: 20, ViLT: 30
BATCH_SIZE=32  
ACCUMULATION_STEPS=4 
CHECKPOINT_INTERVAL=1000

DISTILLATION_LOSS=2e-3 #main_train:2e-3, ablation_no_distill: 0.0
DISTIL_START_POINT=50  #main_train:50 #twotimes: 100
SMALLER_MARGIN=0.005 #FLAVA: main_train: 0.005 | ViLT: main_train:0.01
MARGIN=0.01 #FLAVA: main_train:0.01 | ViLT: main_train: 0.02

# Training Method
TRAINING_ALGO="TAP" #TAP #local #DisentAFL
TRANSFORMER_OPTION="gated" 

#model architecture
EMBED_DIM=768 
MODEL_TYPE='FLAVA' #FLAVA, ViLT
LAYERS=1 #FLAVA:1, ViLT:4
NUM_HEADS=12 #FLAVA:12 #ViLT: 12
MIN_LR=1e-4 #FLAVA:1e-4 #ViLT:1e-4
PEAK_LR=3e-4 #FLAVA:3e-4 #ViLT: 4e-4

#alternative model loading path. USE ONLY FOR ablation_no_distill and twotimes EXP_NAME
USE_ALT_PATH=0 #BY DEFAULT: 0, no_distill and twotimes: 1
ALT_LOAD_PATH="${HOME}/TAP_final/logs/${MODEL_TYPE}/${TRAINING_ALGO}/${SEED}/main_train"

# Create log directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${HOME}/TAP_final/logs/${MODEL_TYPE}/${TRAINING_ALGO}/${SEED}/${EXP_NAME}" 
MODEL_DIR="${HOME}/TAP_final/logs/${MODEL_TYPE}/${TRAINING_ALGO}/${SEED}/${EXP_NAME}" 
mkdir -p ${LOG_DIR}
mkdir -p ${MODEL_DIR}

# Uncomment if you already cached the datasets from HuggingFace, comment if you need to download them
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HOME}/.cache/huggingface"

# use for determinism with operations like dropout
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=${SEED} 

# Run the experiment
CUDA_VISIBLE_DEVICES=$GPU CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python3 ${HOME}/TAP_final/main_training.py \
    --seed ${SEED} \
    --device ${DEVICE} \
    --log_dir ${MODEL_DIR} \
    --num_rounds ${NUM_ROUNDS} \
    --clients_per_round ${CLIENTS_PER_ROUND} \
    --local_rounds ${LOCAL_ROUNDS} \
    --batch_size ${BATCH_SIZE} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --embed_dim ${EMBED_DIM} \
    --training_algo ${TRAINING_ALGO} \
    --transformer_option ${TRANSFORMER_OPTION} \
    --distil_loss_weight ${DISTILLATION_LOSS} \
    --distil_start_point ${DISTIL_START_POINT} \
    --margin ${MARGIN} \
    --smaller_margin ${SMALLER_MARGIN} \
    --model_type ${MODEL_TYPE} \
    --num_layers ${LAYERS} \
    --num_heads ${NUM_HEADS} \
    --min_lr ${MIN_LR} \
    --peak_lr ${PEAK_LR} \
    --use_alt_path ${USE_ALT_PATH} \
    --alternative_postrain_path ${ALT_LOAD_PATH} \
    2>&1 | tee ${LOG_DIR}/${TRAINING_ALGO}experiment_${SEED}.log
echo "Experiment completed. Logs saved to ${LOG_DIR} and models are saved to ${MODEL_DIR}"