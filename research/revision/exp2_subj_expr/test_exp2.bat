@echo off
REM ============================================================================
REM  Revision Exp2: evaluate the Exp2 checkpoint on all four cells.
REM    ref = seen subject / seen expression   (official test frames)
REM    a   = unseen subject / seen expression
REM    b   = seen subject / unseen expression
REM    c   = unseen subject / unseen expression
REM  Results: revision\exp2_subj_expr\results\<cell>\per_sample.csv
REM  Re-running resumes each cell where it stopped.
REM
REM  Run from:  (3dmesh) D:\COURSAI\3Dis> research\revision\exp2_subj_expr\test_exp2.bat
REM ============================================================================
set PYTHONIOENCODING=utf-8

for %%C in (ref a b c) do (
    python scripts\evaluate_benchmark.py ^
        --config_file research\configs\config_diffusion_exp2_subjexpr.cfg ^
        --param x0 --tag depthw_nrm ^
        --bench_dir revision\exp2_subj_expr\benchmark\%%C ^
        --out_dir revision\exp2_subj_expr\results\%%C %*
)
