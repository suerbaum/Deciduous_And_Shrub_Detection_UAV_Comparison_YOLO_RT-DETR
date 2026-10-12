#!/bin/bash
# cnn_summer_yolo_bifpn_2cls_v19_slurm.sh
#SBATCH --job-name=bifpn_su_v19
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz/results/summer_yolo_bifpn_2cls_v19_%j.log
#SBATCH --error=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz/results/summer_yolo_bifpn_2cls_v19_%j.err
DSS=/dss/dssfs04/lwp-dss-0002/pn67pu/pn67pu-dss-0000/ra26fuz
srun --ntasks=1 \
    --container-image=$HOME/nvidia+pytorch+24.01-py3.sqsh \
    --container-mounts=$DSS:$DSS \
    bash -c "pip install 'numpy<2' opencv-python-headless ultralytics -q && sed -i 's/LayerId = cv2.dnn.DictValue/LayerId = None/g' /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py && unset RANK && unset LOCAL_RANK && python $DSS/scripts/cnn_summer_yolo_bifpn_2cls_v19.py"