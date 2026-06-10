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

  if [ "$block_size" -eq 8 ]; then
    prog="./bin/matrixMulN_8"
  elif [ "$block_size" -eq 16 ]; then
    prog="./bin/matrixMulN_16"
  elif [ "$block_size" -eq 32 ]; then
    prog="./bin/matrixMulN_32"
  else
    echo "Invalid block size: $block_size"
    exit 1
  fi

  echo "$prog -n=$thread_work"
}

label_from_params() {
  local thread_work=$1
  local block_size=$2

  echo "tw${thread_work}x${thread_work}_bs${block_size}"
}

run_profiler_for() {
  local thread_work=$1
  local block_size=$2

  local cmd=$(cmd_from_params $thread_work $block_size)
  local label=$(label_from_params $thread_work $block_size)
  local run_dir="$profiler_output_dir/$label"

  mkdir -p "$run_dir"

  nvprof --export-profile "$run_dir/timeline.prof" $cmd
  for metric in "${metrics[@]}"; do
    nvprof --metrics $metric $cmd 2> "$run_dir/$metric.csv" 1> "$run_dir/$metric.txt"
  done

}

for thread_work in 1 2 3 4 5 6; do
  block_size=32
  echo "Running profiler for $(label_from_params $thread_work $block_size) by: $(cmd_from_params $thread_work $block_size)"
  run_profiler_for "$thread_work" "$block_size"
done

for thread_work in 1 2 3 4 6 8 12 16 24; do
  block_size=16
  echo "Running profiler for $(label_from_params $thread_work $block_size) by: $(cmd_from_params $thread_work $block_size)"
  run_profiler_for "$thread_work" "$block_size"
done

for thread_work in 1 2 3 4 6 8 12 16 24 32 64; do
  block_size=8
  echo "Running profiler for $(label_from_params $thread_work $block_size) by: $(cmd_from_params $thread_work $block_size)"
  run_profiler_for "$thread_work" "$block_size"
done