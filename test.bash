# nvprof --metrics <metryki>  --events <zdarzenia> -o <plik.prof> ./bin/matrixMulN

metrics=(
  "gld_transations_per_request",
  "gst_transations_per_request",
  "shared_load_transations_per_request",
  "shared_store_transations_per_request",
  "achieved_occupancy",
  "sm_efficiency",
  "ipc",
  "flop_sp_efficiency",
)

events=(
  "shared_ld_bank_conflict",
  "shared_st_bank_conflict"
)

cmd_with_params() {
  local thread_work=$1
  local block_size=$2
  local registers=$3

  local prog
  if [ $registers -eq 1 ]; then
    prog="./bin/matrixMulN_reg"
  else
    prog="./bin/matrixMulN"
  fi

  echo "$prog -wA 3200 -hA 3200 -wB 3200 -hB 3200 -bs $block_size"
}

cmd_with_params 16 16 1

run_profiler_for() {
  local cmd=$1

  for metric in "${metrics[@]}"; do
    nvprof --metrics $metric -o "profile_${metric}.prof" $cmd
  done

}