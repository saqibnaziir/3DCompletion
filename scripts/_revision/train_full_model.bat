@echo off
REM ============================================================================
REM  Revision exp 1: retrain the FULL paper model on the official COMA split.
REM  Same settings as data/checkpoints_diffusion/predict_x0_depthw_nrm_scratch
REM  (x0 + normalized geodesic depth weight + normal loss 0.01, bs 4, seed 1).
REM
REM  Data   : revision/data   (train 15,182 / val 3,796; test set excluded)
REM  Output : revision/checkpoints/diffusion/predict_x0_depthw_nrm/
REM
REM  Run from:  (3dmesh) D:\COURSAI\3Dis> scripts\_revision\train_full_model.bat
REM  Resume  :  (3dmesh) D:\COURSAI\3Dis> scripts\_revision\train_full_model.bat --resume
REM ============================================================================
set PYTHONIOENCODING=utf-8

python scripts\train_diffusion.py ^
    --config_file config\config_diffusion_revision.cfg ^
    --experiment_name rev_official ^
    --param x0 ^
    --depth_weight --depth_weight_normalize ^
    --lambda_normal 0.01 ^
    --tag depthw_nrm ^
    --epochs 300 --batch_size 4 --seed 1 %*
