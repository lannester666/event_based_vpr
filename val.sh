export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_eo+e.yaml &
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_eo+i.yaml &
wait
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_ie_val.yaml &
