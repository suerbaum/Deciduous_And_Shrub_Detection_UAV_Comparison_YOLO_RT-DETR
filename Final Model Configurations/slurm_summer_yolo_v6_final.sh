#!/bin/bash
# slurm_summer_yolo_v6_final.sh
# YOLOv8m is faster than RT-DETR — 200 epochs batch=16 ~2-3 hours

#SBATCH --job-name=yolo_v6_su_fin
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz/results/yolo_v6_summer_final_%j.log
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz/results/yolo_v6_summer_final_%j.err

DSS=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz

srun --ntasks=1 \
    --container-image=$HOME/nvidia+pytorch+24.01-py3.sqsh \
    --container-mounts=$DSS:$DSS \
    bash -c "pip install 'numpy<2' opencv-python-headless ultralytics -q && \
             pip uninstall thop -y && \
             sed -i 's/LayerId = cv2.dnn.DictValue/LayerId = None/g' \
             /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py && \
             unset RANK && unset LOCAL_RANK && \
             python $DSS/scripts/cnn_summer_yolo_v6_final.py"