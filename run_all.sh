#!/usr/bin/env sh
PY=${PY:-python}
mkdir -p logs
for s in 00_download_ibtracs 01_prepare_inputs 02_baseline_anomalies 03_matching_event_study 04_tables 05_grading_maps_xai 06_grading_figure 07_figures 08_supplementary_inference 09_screen_and_diagnostics 10_bootstrap_pvalues 11_dataset_figures 12_workflow_and_selection_figures; do
  echo "== $s"
  if ! $PY scripts/$s.py > logs/$s.log 2>&1; then
    echo "stage $s failed; last lines of logs/$s.log:"
    tail -n 20 logs/$s.log
    exit 1
  fi
  tail -n 2 logs/$s.log
done
echo "All analysis stages completed."
