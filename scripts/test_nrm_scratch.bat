@echo off
REM ===========================================================================
REM  Score the FROM-SCRATCH normal-loss arm  (predict_x0_depthw_nrm_scratch)
REM  Uses the best-eval checkpoint (diffusion_denoiser_ema.pt = epoch 269),
REM  which is what every scorer loads by default and how the other arms were
REM  scored too.
REM
REM  Run from the project root in the activated conda `3dmesh` env:
REM      (3dmesh) D:\COURSAI\3Dis> scripts\test_nrm_scratch.bat
REM ===========================================================================

setlocal
set PYTHONUTF8=1
set TEST_DIR=data/completion_test_samples/test_set
set TAG=depthw_nrm_scratch

echo.
echo ============================================================
echo  [1/4] Completing the 20 test meshes
echo ============================================================
python scripts\infer_diffusion.py --data COMA --param x0 --tag %TAG% ^
    --eval_test --test_dir %TEST_DIR% --out_suffix _x0_depthw_nrm_scratch
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  [2/4] Single-draw position + normal table vs the other arms
echo ============================================================
python scripts\compare_arms.py ^
    --arm "x0_depthw:%TEST_DIR%/completed_diffusion_x0_depthw" ^
    --arm "nrm316:%TEST_DIR%/completed_diffusion_x0_depthw_nrm316" ^
    --arm "nrm_scratch:%TEST_DIR%/completed_diffusion_x0_depthw_nrm_scratch"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  [3/4] Sampling-floor diagnostic, K=10
echo         ^-- this produces the numbers you report
echo ============================================================
python scripts\diag_sampling_floor.py --data COMA --param x0 --tag %TAG% ^
    --test_dir %TEST_DIR% --K 10 ^
    --out_dir %TEST_DIR%/diag_sampling_floor_nrm_scratch
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  [4/4] Paired stats vs parent / ctrl / nrm316
echo ============================================================
python scripts\paired_arms.py ^
    --arm "parent:%TEST_DIR%/diag_sampling_floor" ^
    --arm "ctrl:%TEST_DIR%/diag_sampling_floor_ctrl" ^
    --arm "nrm316:%TEST_DIR%/diag_sampling_floor_nrm316" ^
    --arm "scratch:%TEST_DIR%/diag_sampling_floor_nrm_scratch" ^
    --ref parent --out_csv %TEST_DIR%/paired_nrm_scratch.csv
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  DONE. Beat these (depthw parent, mean-of-K over 10 draws):
echo      normals hole 3.497 deg   seam 0.816 deg   position 1.017 mm
echo  The fine-tuned nrm316 arm reached 3.403 / 0.785 / 0.969.
echo ============================================================
goto :eof

:failed
echo.
echo *** FAILED at the step above -- fix it before running the rest. ***
exit /b 1
