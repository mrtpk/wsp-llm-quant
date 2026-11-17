#!/bin/bash
# start_queues.sh QUEUEFILE [SLOTS=2] [GPUS="1"]
# Start SLOTS queue workers on each GPU, all sharing one queue file (mkdir locks prevent double claims).
# Policy (user, 2026-11-11): one GPU (GPU 1) for wsp, at most 2 jobs (don't overload the icpc host).
Q=$1; S=${2:-2}; GPUS=${3:-"1"}
cd "$(dirname "$0")"
for g in $GPUS; do for k in $(seq 1 $S); do
  tmux new-session -d -s "q-${Q%.tsv}-g${g}s${k}" "./queue.sh $g $Q 2>&1 | tee -a /models/tmp/wsp/logs/queue_g${g}.log"
  sleep 2
done; done
tmux ls | grep "q-${Q%.tsv}"
