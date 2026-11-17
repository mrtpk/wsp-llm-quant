#!/bin/bash
# Round 3 phase A chain (2026-11-15). GPU 1, one job at a time. Stops on any failure.
set -e
L=/models/tmp/wsp2/logs/p3a
while docker ps --format '{{.Names}}' | grep -q '^wsp2-g1-'; do sleep 60; done     # wait for the sensitivity job
grep -q "sens done" $L/sens32.log || { echo "sens did not finish" >> $L/chain.log; exit 1; }
./drun.sh 1 bash -c "cd /code && python -m r2.alloc alloc" > $L/alloc.log 2>&1
grep -q "^DONE" $L/alloc.log || { echo "alloc failed" >> $L/chain.log; exit 1; }
for s in 0 1 2 3; do
  ./drun.sh 1 bash -c "cd /code && python -m r2.alloc run mix_asym_W3_late34 $s" > $L/ctl_late34_s$s.log 2>&1
  grep -q "^DONE" $L/ctl_late34_s$s.log || { echo "late34 s$s failed" >> $L/chain.log; exit 1; }
done
# secondary only if it can start before 23:15 UTC (phase-A budget <= ~15 GPU-hours from 12:38)
if [ $(date -u +%H%M) -lt 2315 ] && [ $(date -u +%d) -eq 15 ]; then
  for s in 0 1 2 3; do
    ./drun.sh 1 bash -c "cd /code && python -m r2.alloc run mix_asym_W3g64_late44 $s" > $L/ctl_late44_s$s.log 2>&1
    grep -q "^DONE" $L/ctl_late44_s$s.log || { echo "late44 s$s failed" >> $L/chain.log; exit 1; }
  done
else
  echo "secondary dropped: budget (start after 23:15 UTC)" >> $L/chain.log
fi
echo "chain done $(date -u)" >> $L/chain.log
