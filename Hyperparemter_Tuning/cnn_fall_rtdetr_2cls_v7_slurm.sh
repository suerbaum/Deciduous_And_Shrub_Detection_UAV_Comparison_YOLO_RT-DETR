#!/bin/bash
# cnn_fall_rtdetr_2cls_v7_slurm.sh
#
# SLURM job script for fall RT-DETR v7

#SBATCH --job-name=rtdetr_fal_v7
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz/results/fall_rtdetr_2cls_v7_%j.log
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz/results/fall_rtdetr_2cls_v7_%j.err

DSS=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz

srun --ntasks=1 \
    --container-image=$HOME/nvidia+pytorch+24.01-py3.sqsh \
    --container-mounts=$DSS:$DSS \
    bash -c "pip install 'numpy<2' opencv-python-headless ultralytics -q && sed -i 's/LayerId = cv2.dnn.DictValue/LayerId = None/g' /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py && unset RANK && unset LOCAL_RANK && python $DSS/scripts/cnn_fall_rtdetr_2cls_v7.py"