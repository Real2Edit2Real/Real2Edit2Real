# Parse parameters
GPU_ID=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | \
         nl -v 0 | \
         sort -nrk 2 | \
         cut -f 1 | \
         head -n 1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
echo Choose GPU ID: $CUDA_VISIBLE_DEVICES

config_path=configs/mug_to_box_1654490_bs60.yaml
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-path)
            config_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

config_dir=$(realpath "$(dirname "$config_path")")
config_name=$(basename "$config_path")

python tools/generate_demo.py --config-path $config_dir --config-name $config_name