# DualGenerator:Information Interaction-based Generative Network for Point Cloud Completion


### MVP Completion Dataset
<!-- Download the MVP completion dataset by the following commands:
```
cd data; sh download_data.sh
``` -->
Download the MVP completion dataset [Google Drive](https://drive.google.com/drive/folders/1XxZ4M_dOB3_OG1J6PnpNvrGTie5X9Vk_) or [百度网盘](https://pan.baidu.com/s/18pli79KSGGsWQ8FPiSW9qg)&nbsp;&nbsp;(code: p364) to the folder "data".

The data structure will be:
```
data
├── MVP_Train_CP.h5
|    ├── incomplete_pcds (62400, 2048, 3)
|    ├── complete_pcds (2400, 2048, 3)
|    └── labels (62400,)
├── MVP_Test_CP.h5
|    ├── incomplete_pcds (41600, 2048, 3)
|    ├── complete_pcds (1600, 2048, 3)
|    └── labels (41600,)
```
### Installation
Install Anaconda, and then use the following command:
```
conda create -n mvp python=3.7 -y
conda activate mvp
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch -y

# setup completion
cd completion
pip install -r requirements.txt


cd ../utils/mm3d_pn2/
sh setup.sh

pip install -v -e . 

cd ../../
```
### Usage
+ To train a model: run `python train.py -c ./cfgs/DualG.yaml`
+ To test a model: run `python test.py -c ./cfgs/DualG.yaml`
+ `run_train.sh` and `run_test.sh` are provided for SLURM users. 
+ Different partial point clouds for the same CAD Model:



