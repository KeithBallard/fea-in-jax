#!/bin/bash

# Configuration
# METHODS=("JAX_HASH" "JZTREE")
METHODS=("JAX_HASH" "JZTREE" "WARP_NODE_CLOUD" "WARP_ROD" "SCIPY_KDTREE" "CUPY_KDTREE")
TOWS=("1" "1_2" "2_3_2")
DX_LEVELS=(0 1 2 3)
DATE_STR=$(date +%Y%b%d)
JSON_OUT="ContactScaling/${DATE_STR}_angle_interlock_scaling_results.json"
SELF_ADJ=50000

# Compute total runs
TOTAL_RUNS=$((${#TOWS[@]} * ${#DX_LEVELS[@]} * ${#METHODS[@]}))
RUN_COUNT=0
FAILED_RUNS=0

echo "======================================================================"
echo " Starting Angle Interlock Contact Scaling Study ($TOTAL_RUNS total micro-runs)"
echo " Results file (via get_output): output/$JSON_OUT"
echo "======================================================================"

for tow in "${TOWS[@]}"; do
  for n in "${DX_LEVELS[@]}"; do
    for method in "${METHODS[@]}"; do
      RUN_COUNT=$((RUN_COUNT + 1))
      
      echo ""
      echo "----------------------------------------------------------------------"
      echo "On $RUN_COUNT of $TOTAL_RUNS | Method: $method | Tow: $tow | dX Level: $n"
      echo "----------------------------------------------------------------------"
      
      python timings/contact/run_single_benchmark_angle_interlock.py \
        --method "$method" \
        --n_dx "$n" \
        --tow "$tow" \
        --self_adj "$SELF_ADJ" \
        --json_out "$JSON_OUT"
      
      EXIT_CODE=$?
      if [ $EXIT_CODE -eq 0 ]; then
        echo "--> [STATUS] Run $RUN_COUNT of $TOTAL_RUNS ($method) COMPLETED."
      else
        echo "--> [FAILED] Run $RUN_COUNT of $TOTAL_RUNS ($method) FAILED with exit code $EXIT_CODE."
        FAILED_RUNS=$((FAILED_RUNS + 1))
      fi


      
      # 3-second GPU memory cleanup & thermal cooldown pause
      sleep 3
    done
  done
done

echo ""
echo "======================================================================"
echo " All $TOTAL_RUNS benchmarks processed ($FAILED_RUNS failures)."
echo " Results saved via get_output in output/$JSON_OUT"
echo "======================================================================"
