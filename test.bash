# nvprof --metrics <metryki>  --events <zdarzenia> -o <plik.prof> ./bin/matrixMulN

profiler_output_dir="profiler_outputs"
mkdir -p "$profiler_output_dir"

metrics=(
  "gld_transations_per_request"
  "gst_transations_per_request"
  "shared_load_transations_per_request"
  "shared_store_transations_per_request"
  "achieved_occupancy"
  "sm_efficiency"
  "ipc"
  "flop_sp_efficiency"
)

events=(
  "shared_ld_bank_conflict"
  "shared_st_bank_conflict"
)

cmd_from_params() {
  local thread_work=$1
  local block_size=$2
  local registers=$3

  local prog
  if [ $registers -eq 1 ]; then
    prog="./bin/matrixMulN_reg"
  else
    prog="./bin/matrixMulN"
  fi

  echo "$prog -wA=3200 -hA=3200 -wB=3200 -hB=3200 -bs=$block_size"
}

label_from_params() {
  local thread_work=$1
  local block_size=$2
  local registers=$3

  local reg_label
  if [ $registers -eq 1 ]; then
    reg_label="reg"
  else
    reg_label="noreg"
  fi

  echo "tw${thread_work}_bs${block_size}_${reg_label}"
}

run_profiler_for() {
  local thread_work=$1
  local block_size=$2
  local registers=$3

  local cmd=$(cmd_from_params $thread_work $block_size $registers)
  local label=$(label_from_params $thread_work $block_size $registers)
  local run_dir="$profiler_output_dir/$label"

  mkdir -p "$run_dir"

  nvprof --export-profile "$run_dir/timeline.prof" $cmd
  for metric in "${metrics[@]}"; do
    nvprof --metrics $metric -o "$run_dir/profile_${metric}.prof" $cmd
  done

}

for thread_work in 1 2 3 4 5 6; do
  for block_size in 8 16 32; do
    for registers in 0 1; do

      
      echo "Running profiler for $(cmd_from_params $thread_work $block_size $registers)"
      run_profiler_for "$thread_work" "$block_size" "$registers"

    done
  done
done