WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo $WORLD_SIZE $gpu_num $RANK $MASTER_ADDR $MASTER_PORT

torchrun --nnodes=$WORLD_SIZE \
         --nproc_per_node=$gpu_num \
         --node_rank=$RANK \
         --master-addr $MASTER_ADDR \
         --master-port $MASTER_PORT \
         $@