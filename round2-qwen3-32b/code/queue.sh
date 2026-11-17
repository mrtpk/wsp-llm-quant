#!/bin/bash
# queue.sh GPU QUEUEFILE : run jobs from QUEUEFILE on one GPU, in order, unattended.
# Line format:  NAME<TAB>COMMAND   (COMMAND runs inside the container via drun.sh)
# A job is done when /models/tmp/wsp2/results/NAME.done exists. A job is claimed by
# creating NAME.lock atomically (mkdir), so several GPUs can share one queue file.
# The file is re-read after each job, so lines can be appended while it runs.
G=$1; Q=$2
R=/models/tmp/wsp2/results; L=/models/tmp/wsp2/logs
cd "$(dirname "$0")"
while true; do
  job=""
  while IFS=$'\t' read -r name cmd; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -e $R/$name.done ]] && continue
    if mkdir $R/$name.lock 2>/dev/null; then job="$name"; jcmd="$cmd"; break; fi
  done < "$Q"
  [[ -z "$job" ]] && { echo "$(date -u +%FT%TZ) gpu$G queue empty"; exit 0; }
  echo "$(date -u +%FT%TZ) gpu$G START $job"
  ./drun.sh $G bash -c "$jcmd" > $L/$job.log 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then touch $R/$job.done; echo "$(date -u +%FT%TZ) gpu$G DONE $job"
  else echo "$(date -u +%FT%TZ) gpu$G FAIL $job rc=$rc"; mv $R/$job.lock $R/$job.failed.$(date +%s); fi
done
