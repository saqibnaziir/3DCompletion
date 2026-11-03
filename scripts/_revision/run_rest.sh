#!/usr/bin/env bash
# Queue the remaining GPU measurements so they run strictly one after another
# (concurrent jobs would contaminate the runtime numbers).
set -u
PY=/c/Users/loanr/anaconda3/envs/3dmesh/python.exe
cd /d/COURSAI/3Dis

# 1. wait for the depth-decile run to drop its summary
until [ -f data/_revision/depth_deciles_summary.json ]; do sleep 20; done
echo "=== depth deciles done, starting K sweep ==="
sleep 5

# 2. K sweep (part 3)
"$PY" -u scripts/_revision/rev_k_sweep.py > /tmp/rev3.log 2>&1
echo "=== K sweep exit=$? ==="

# 3. FaceCom per-mesh wall clock (part 2 baseline)
cd Compattators/FaceCom
"$PY" -u test_facecom_coma.py --units mm --quiet > /tmp/rev_facecom.log 2>&1
echo "=== facecom exit=$? ==="
echo ALLDONE
