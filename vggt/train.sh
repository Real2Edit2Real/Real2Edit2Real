#!/bin/bash

CONFIG_PATH="./config"  
TRAIN_CONFIG_PATH="train_config.yaml" 
MAX_IMAGE_PER_GPU=6
NUM_WORKERS=16          
MAX_ITERATIONS=150000   
ACCU_STEP=4  
LOG_STEP=400             
EVAL_STEP=10000           
LR=0.00002                
LR_BACKBONE=0.00002      
WEIGHT_DECAY=0.01     
CAMERA_WEIGHT=10 
DEPTH_WEIGHT=1
POINT_WEIGHT=1
GRAD_WEIGHT=1                
SAVE_DIR=./checkpoints/vggt_v1
WORLD_SIZE=8      
DIST_URL="env://"     

torchrun \
    --nproc_per_node=$WORLD_SIZE \
    train.py \
    --config_path $CONFIG_PATH \
    --train_config_path $TRAIN_CONFIG_PATH \
    --max_image_per_gpu $MAX_IMAGE_PER_GPU \
    --num_workers $NUM_WORKERS \
    --max_iterations $MAX_ITERATIONS \
    --accu_step $ACCU_STEP \
    --log_step $LOG_STEP \
    --eval_step $EVAL_STEP \
    --lr $LR \
    --lr_backbone $LR_BACKBONE \
    --camera_loss_weight $CAMERA_WEIGHT \
    --depth_loss_weight $DEPTH_WEIGHT \
    --point_loss_weight $POINT_WEIGHT \
    --grad_loss_weight $GRAD_WEIGHT \
    --weight_decay $WEIGHT_DECAY \
    --log_dir $SAVE_DIR \
    --save_dir $SAVE_DIR \
    --world_size $WORLD_SIZE \
    2>&1 | tee -a $SAVE_DIR/record.log \
    