#!/bin/bash
# Emergence lab — overnight run. All output lands in results/ and logs/.
# Each experiment logs to its own file and failures do not stop the night.
cd "$(dirname "$0")"
PY=/home/qwertylab/projects/thesis/proyecto-dirigido/feature_emergence/.venv/bin/python
mkdir -p results logs

echo "=== lab start $(date) ==="
for exp in exp_a_sine exp_b_grokking exp_c_double; do
  echo "=== $exp start $(date) ==="
  "$PY" "$exp.py" > "logs/$exp.log" 2>&1 \
    && echo "=== $exp OK $(date) ===" \
    || echo "=== $exp FAILED (see logs/$exp.log) $(date) ==="
done
echo "=== lab done $(date) ==="
