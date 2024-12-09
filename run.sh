#!/bin/bash​
#$ -cwd              # execute the job from the current directory​
#$ -pe smp 12          # 8 cores per GPU​
#$ -l h_rt=240:0:0    # 240 hours runtime​
#$ -l h_vmem=7.5G      # 11G RAM per core​
#$ -l gpu=1           # request 1 GPU​
#$ -l cluster=andrena # use the Andrena nodes


module load anaconda3
conda activate srvenv
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

python -c 'import torch; print(torch.cuda.is_available())'
cd /data/home/xaw004/Codes/MICCAI2025/MINet/experimental/MINet
PYTHONPATH="/data/home/xaw004/Codes/MICCAI2025/MINet" python3 train.py
#cd /data/home/xaw004/Codes/DEXA_Clustering


# install from requirements.txt
# conda install --yes --file requirements.txt

