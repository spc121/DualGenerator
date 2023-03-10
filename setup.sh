#!/bin/bash

############## make sure you can write in \tmp; Or you should set TORCH_EXTENSIONS_DIR
# e.g. export TORCH_EXTENSIONS_DIR=/mnt/lustre/$YourUserName$/tmp

conda create -n mvp python=3.7 -y
conda activate mvp
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch -y

cd completion
pip install -r requirements.txt


cd ../utils/mm3d_pn2/
sh setup.sh

pip install -v -e . 

cd ../../
