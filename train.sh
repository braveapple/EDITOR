set -e

source activate EDITOR    
export PYTHONPATH=$PYTHONPATH:/mnt/disk/wpy_data/code/shell    
export CUDA_VISIBLE_DEVICES=$(python -c 'from nv import get_one_gpu_loop; print(get_one_gpu_loop(min_mem=30, max_day_time=2))')

exp_01() {
    DATASET=${1}
    OUTPUT_DIR=/mnt/disk/wpy_data/experiments/multi_modality_object_reidentification/EDITOR/${DATASET}/ViT_BatchSize128
    mkdir -p ${OUTPUT_DIR} && cp train.sh ${OUTPUT_DIR}

    python train.py --config_file configs/${DATASET}/EDITOR.yml \
        SOLVER.IMS_PER_BATCH 128 \
        OUTPUT_DIR ${OUTPUT_DIR}
}

# exp_02() {
#     DATASET=${1}
#     OUTPUT_DIR=/mnt/disk/wpy_data/experiments/multi_modality_object_reidentification/EDITOR/${DATASET}/vit_base_distri
#     mkdir -p ${OUTPUT_DIR} && cp train.sh ${OUTPUT_DIR}

#     python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train.py \
#         --config_file configs/${DATASET}/EDITOR.yml \
#         MODEL.DIST_TRAIN True \
#         SOLVER.IMS_PER_BATCH 128 \
#         OUTPUT_DIR ${OUTPUT_DIR}
# }

exp_01 RGBNT100
exp_01 RGBNT201    
exp_01 MSVR310