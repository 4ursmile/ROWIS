#!/bin/bash
python tools/train_net.py --config-file configs/OWIS/sparse_inst_r50_T1.yaml --num-gpus 2 --resume MODEL.WEIGHTS pretrained_models/model_final_t0_raw.pth

python tools/train_net.py --config-file configs/OWIS/sparse_inst_r50_T2.yaml --num-gpus 2 --resume MODEL.WEIGHTS output/t1/model_final.pth

python tools/train_net.py --config-file configs/OWIS/sparse_inst_r50_T3.yaml --num-gpus 2 --resume MODEL.WEIGHTS output/t2/model_final.pth

python tools/train_net.py --config-file configs/OWIS/sparse_inst_r50_T4.yaml --num-gpus 2 --resume MODEL.WEIGHTS output/t3/model_final.pth