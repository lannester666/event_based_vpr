export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_i.yaml &
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_e.yaml &
export CUDA_VISIBLE_DEVICES=0 ;python3 main.py --config configs/resnet18_ie.yaml &
wait
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_ie_bi.yaml &
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_ie_val.yaml &
export CUDA_VISIBLE_DEVICES=0 ;python3 main.py --config configs/resnet18_e_original.yaml &
wait
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_eo+i.yaml &
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_eo+i_bi.yaml &
export CUDA_VISIBLE_DEVICES=0 ;python3 main.py --config configs/resnet18_eo+i_inference.yaml &
wait
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_e_denoise.yaml &
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_ed+i.yaml &
export CUDA_VISIBLE_DEVICES=0 ;python3 main.py --config configs/resnet18_ed+i_bi.yaml &
wait
export CUDA_VISIBLE_DEVICES=1 ;python3 main.py --config configs/resnet18_ed+i_inference.yaml &