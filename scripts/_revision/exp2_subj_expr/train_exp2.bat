@echo off
REM ============================================================================
REM  Revision Exp2: (subject, expression) split. Same model and settings as
REM  scripts\_revision\train_full_model.bat (R1); only the data differs.
REM
REM  Data   : revision\exp2_subj_expr\data   (S_train x E_train only)
REM  Output : revision\exp2_subj_expr\checkpoints\diffusion\predict_x0_depthw_nrm\
REM
REM  Before the first run (once):
REM    python scripts\_revision\exp2_subj_expr\make_split.py
REM    python scripts\_revision\process_official_split.py --config_file config\config_diffusion_exp2_subjexpr.cfg
REM    python scripts\_revision\exp2_subj_expr\check_processed.py
REM
REM  Run from:  (3dmesh) D:\COURSAI\3Dis> scripts\_revision\exp2_subj_expr\train_exp2.bat
REM  Resume  :  (3dmesh) D:\COURSAI\3Dis> scripts\_revision\exp2_subj_expr\train_exp2.bat --resume
REM ============================================================================
set PYTHONIOENCODING=utf-8

python scripts\train_diffusion.py ^
    --config_file config\config_diffusion_exp2_subjexpr.cfg ^
    --experiment_name exp2_subjexpr ^
    --param x0 ^
    --depth_weight --depth_weight_normalize ^
    --lambda_normal 0.01 ^
    --tag depthw_nrm ^
    --epochs 300 --batch_size 4 --seed 1 %*
