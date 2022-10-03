## Reproduced ScanNet Flow Evaluation Results
|       | Epe    | Out  |
|-------|-------|------|
| DKMv2-Reproduction | 5.32  | 0.386 |
| DKMv2 | **4.46**  | **0.295** |
* I benchmarked on the Flow to avoid disturbance from confidence maps. 
* Both metrics Epe & Out are the lower the better.
* Metric Out is off too much


## Reproduced Megadepth1500 Results
|       | @5    | @10  | @20  |
|-------|-------|------|------|
| DKMv2-Reproduction| 46.2  | 61.9 | 73.8 |
| DKMv2 | **56.8**  | **72.3** | **83.2** |
* The results are reproduced using provided validation scripts
## Remakrs
* I can reproduce Test Results with released checkpoints
* I installed Cupy, enabling local_corr.py
* I did not freeze BatchNormalization during Reproduction
* I am running a reproduction experiments following the exactly same environments provided. (Not Have Results Yet)
* The checkpoints & Tensorboad records are put in the [Google Drive Link](https://drive.google.com/drive/folders/1Aqdmzaw7iLg884zpzDbZiajfnPZJwvZA?usp=sharing).
* I put all my environments in two requirements_conda.txt & requirements_pip.txt (from conda & pip seperataly)

## May I have......
* May I have the environment file requirements_conda.txt & requirements_pip.txt on your end?
``` bash
python -m pip freeze > requirements_pip.txt
conda list -e > requirements_conda.txt
```
* May I have the tensorboard file or other file that records losses & learning rate?
* May I have Pickle File of One Complete Iteration? For reference:
    1. Batch Input
    2. Computed Loss: (Depth Loss & Confidence Loss) 
    3. Initialized Model Parameters

## Create Environment
I created the environment manually. The machine uses a cuda version 11.3
``` bash
conda create -n DKMReProduction python=3.6
conda activate DKMReProduction
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install einops
pip install opencv-python
pip install kornia
pip install albumentations
pip install loguru
pip install tqdm
pip install matplotlib
pip install h5py
pip install tensorboard
pip install cupy-cuda113
```


## Reproduction Command
* Training
``` bash
CUDA_VISIBLE_DEVICES=0,1,2 python train/train_mega_dkm_nnparallel_wvalidation.py --experiment_name reproduce_dkm_megadepth_nnparallel_wval_bz36_09282022 --gpus 3 \\
--data_root /scratch1/zhusheng/EMAwareFlow/MegaDepth --num_workers 48 \\
--batch_size 36 --relfrmin_eval 0 1 2 3 4 --downscale \\
--scannetroot /scratch1/zhusheng/EMAwareFlow/scannet_test_organized
```
* Validation Reproduced
``` bash
CUDA_VISIBLE_DEVICES=0 python train/train_mega_dkm_nnparallel_wvalidation.py --experiment_name reproduce_dkm_megadepth_nnparallel_wval_bz36_09282022 --gpus 1 \\
--data_root /scratch1/zhusheng/EMAwareFlow/MegaDepth --num_workers 48 \\
--batch_size 36 --relfrmin_eval 0 1 --downscale \\
--scannetroot /scratch1/zhusheng/EMAwareFlow/scannet_test_organized \\
--eval_only --restore_ckpt reproduce_dkm_megadepth_nnparallel_wval_bz36_09282022/minimal_outlier_scannet.pth
```