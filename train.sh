#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=train_ctran_bcepoly2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0005
#SBATCH --output=train_ctran_bcepoly2.%j.out
#SBATCH --error=train_ctran_bcepoly2.%j.err

module purge
module load gcc/9.3
module load python/3.9.6
module load miniconda/3
module load cuda/11.3

pip install -q timm
pip install -q einops
pip install -q nltk
pip install -q pillow
pip install -q numpy
pip install -q torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -qU scikit-learn
pip install -q pandas
pip install -q tensorboard
pip install -qU albumentations
pip install -q scikit-multilearn
pip install -qU iterative-stratification

#!python /kaggle/working/C-Tran/main.py --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3 --dataset 'custom' --use_lmt --grad_ac_step 2 --dataroot /kaggle/input/fyp-dataset-list --results_dir /kaggle/working/  --loss bce_poly --poly_eps 2.0 --img_size 384 --backbone 'densenet' --name 'densenet_bce_poly_2' --run_platform kaggle
#!python /kaggle/working/C-Tran/main.py --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3 --dataset 'custom' --use_lmt --grad_ac_step 2 --dataroot /kaggle/input/fyp-dataset-list --results_dir /kaggle/working/  --loss bce_poly --poly_eps 2.0 --img_size 384 --backbone 'densenet' --name 'densenet_bce_poly_2' --run_platform kaggle
#python main.py --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataset 'custom' --use_lmt --grad_ac_step 2 --dataroot /home/kunet.ae/100058256/datasets/ --results_dir /home/kunet.ae/100058256/codes/trained_models/c_tran/  --loss bce_poly --poly_eps 2.0 --img_size 384 --backbone 'densenet' --name 'densenet_bce_poly_2' --run_platform local_run