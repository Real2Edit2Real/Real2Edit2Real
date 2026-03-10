REQUESTED_GPUS=${REQUESTED_GPUS:-1}

# Check nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Exiting." >&2
    return 1 # Use return to ensure exit in source mode
fi

# 1. Get all GPU information
# Format: index,memory.used
gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$gpu_info" ]; then
    echo "❌ Error: Failed to get GPU info." >&2
    return 1
fi

total_gpus=$(echo "$gpu_info" | wc -l)

# 2. Determine final count
final_count=$total_gpus
if [ "$REQUESTED_GPUS" -gt 0 ]; then
    if [ "$REQUESTED_GPUS" -le "$total_gpus" ]; then
        final_count=$REQUESTED_GPUS
    else
        echo "⚠️ Warning: Requested $REQUESTED_GPUS, using all $total_gpus." >&2
    fi
fi

if [ "$final_count" -eq 0 ]; then
    echo "❌ Error: No GPUs available." >&2
    return 1
fi

# 3. Sort, select and format
selected_gpus=$(
    echo "$gpu_info" |
    sort -t, -k2 -n |
    head -n "$final_count" |
    cut -d, -f1 |
    tr '\n' ',' |
    sed 's/,$//'
)
echo "Selected GPUs: $selected_gpus"
IFS=',' read -ra GPU_ARRAY <<< "$selected_gpus"
gpu_count=${#GPU_ARRAY[@]}

config_path=configs/mug_to_box_1654490_bs60.yaml
save_dir=checkpoints/video_generation_checkpoint
model_id=checkpoints/Cosmos-Predict2-2B-Video2World
save_video="--save_video"
save_video_for_all_inputs=""
stp=6
debug=""
scale=1.0
chunk=25
demo_num=200
depth_canny_type=npz
suffix=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-path)
            config_path="$2"
            shift 2
            ;;
        --save_dir | -s)
            save_dir="$2"
            shift 2
            ;;
        --model_id | -m)
            model_id="$2"
            shift 2
            ;;
        --save_video | -sv)
            save_video="--save_video"
            shift 1
            ;;
        --save_video_for_all_inputs | -svi)
            save_video_for_all_inputs="--save_video_for_all_inputs"
            shift 1
            ;;
        --steps | -stp)
            stp="$2"
            shift 2
            ;;
        --scale | -sc)
            scale="$2"
            shift 2
            ;;
        --debug | -d)
            debug="--debug"
            shift 1
            ;;
        --chunk | -c)
            chunk="$2"
            shift 2
            ;;
        --demo_num | -dn)
            demo_num="$2"
            shift 2
            ;;
        --depth_canny_type)
            depth_canny_type="$2"
            shift 2
            ;;
        --suffix)
            suffix="$2"
            shift 2
            ;;
    esac
done
input_root=$(dirname $(python -c "from omegaconf import OmegaConf; conf = OmegaConf.load('$config_path'); print(conf.output_root)"))
relative_rank=0
for single_gpu_index in "${GPU_ARRAY[@]}"; do
    (
        echo Sending command to "$single_gpu_index", relative_rank is $relative_rank
        # 1. Isolate GPU: the current process can only see the assigned GPU
        export CUDA_VISIBLE_DEVICES=$single_gpu_index
        if [[ -z "$suffix" ]]; then
            python videogen/scripts/infer_action_depth_canny_cosmos2_multigpu.py -r $relative_rank -ng $gpu_count -i $input_root -s $save_dir -m $model_id --steps $stp --scale $scale --chunk $chunk --demo_num $demo_num --depth_canny_type $depth_canny_type $save_video $save_video_for_all_inputs $debug
        else
            python videogen/scripts/infer_action_depth_canny_cosmos2_multigpu.py -r $relative_rank -ng $gpu_count -i $input_root -s $save_dir -m $model_id --steps $stp --scale $scale --chunk $chunk --demo_num $demo_num --depth_canny_type $depth_canny_type --suffix $suffix $save_video $save_video_for_all_inputs $debug
        fi
    ) &  # Parallel execution in the background
    relative_rank=$((relative_rank + 1))
done
wait