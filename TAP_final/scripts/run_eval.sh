#!/bin/bash

# Experiment configuration
EXP_NAME="evaluation" # "evaluation"  #"ablation_no_distill_eval" #"twotimes_eval"
SAVED_EXP="main_train" #"main_train" #"ablation_no_distill" #"twotimes"
SEED=41 #41-44
DEVICE="cuda"  # or "cpu"
GPU=0

BATCH_SIZE=32 
NUM_CLIENTS=30 

# Model parameters
EMBED_DIM=768 

# Evaluation Method
TRAINING_ALGO="TAP" #TAP #local #DisentAFL
TRANSFORMER_OPTION="gated"  

#model architecture
MODEL_TYPE='ViLT' #'FLAVA', 'ViLT'
LAYERS=4 #FLAVA:1, ViLT: 4
NUM_HEADS=12 #FLAVA:12, ViLT:12

SAVE_DIR="${HOME}/TAP_final/evaluation/results/${MODEL_TYPE}/${TRAINING_ALGO}/${SEED}/${EXP_NAME}"
LOG_DIR="${HOME}/TAP_final/logs/${MODEL_TYPE}/${TRAINING_ALGO}/${SEED}/${SAVED_EXP}"
mkdir -p ${LOG_DIR}

# Uncomment if you already cached the datasets, comment if you need to download them
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HOME}/.cache/huggingface"

# use for determinism with operations like dropout
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=${SEED} 

# Run the experiment
CUDA_VISIBLE_DEVICES=$GPU CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python3 ${HOME}/TAP_final/evaluate.py \
    --seed ${SEED} \
    --device ${DEVICE} \
    --log_dir ${LOG_DIR} \
    --save_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_clients ${NUM_CLIENTS} \
    --embed_dim ${EMBED_DIM} \
    --training_algo ${TRAINING_ALGO} \
    --transformer_option ${TRANSFORMER_OPTION} \
    --model_type ${MODEL_TYPE} \
    --num_layers ${LAYERS} \
    --num_heads ${NUM_HEADS} \
    2>&1 | tee ${LOG_DIR}/${TRAINING_ALGO}eval_${SEED}.log

echo "Evaluation completed."