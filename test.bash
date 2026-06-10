# nvprof --metrics <metryki>  --events <zdarzenia> -o <plik.prof> ./bin/matrixMulN

profiler_output_dir="profiler_outputs"
mkdir -p "$profiler_output_dir"

CUDA_VISIBLE_DEVICES=1

metrics=(
  "achieved_occupancy"
  "sm_efficiency"
  "ipc"
  "flop_sp_efficiency"
  "flop_dp_efficiency"
  "flop_count_sp"
  "dram_read_transactions"
  "dram_write_transactions"
  "shared_load_transactions"
  "shared_store_transactions"
  "gld_transactions"
  "gst_transactions"
  "dram_read_bytes"
)

events=(
  "shared_ld_bank_conflict"
  "shared_st_bank_conflict"
)

cmd_from_params() {
  local thread_work=$1
  local block_size=$2

  
  if [ "$block_size" -eq 16 ]; then
    prog="./bin/macierz_16"
  elif [ "$block_size" -eq 32 ]; then
    prog="./bin/macierz_32"
  # elif [ "$block_size" -eq 8 ]; then
  #   prog="./bin/macierz_8"
  else
    echo "Invalid block size: $block_size"
    exit 1
  fi

  echo "$prog -n=$thread_work"
}

label_from_params() {
  local thread_work=$1
  local block_size=$2

  echo "bs${block_size}_tw${thread_work}x${thread_work}"
}

run_profiler_for() {
  local thread_work=$1
  local block_size=$2

  local cmd=$(cmd_from_params $thread_work $block_size)

  local run_dir="$profiler_output_dir/bs$block_size/n$thread_work"

  mkdir -p "$run_dir"

  $cmd > "$run_dir/time.txt"
  # nvprof --export-profile "$run_dir/timeline.prof" $cmd
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

# for thread_work in 1 2 3 4 6 8 12 16 24 32 64; do
#   block_size=8
#   echo "Running profiler for $(label_from_params $thread_work $block_size) by: $(cmd_from_params $thread_work $block_size)"
#   run_profiler_for "$thread_work" "$block_size"
# done