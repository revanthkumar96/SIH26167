#!/bin/bash
# What the Stage B run is doing, in one screen. Lives on the box so the watcher
# does not have to send a quoted script over SSH on every tick -- passing shell
# through two layers of quoting is how a status tool starts reporting its own
# syntax errors instead of the run's state.
L=/workspace/logs/stage-b.log

echo "--- STAGE ---"
grep "^###" "$L" 2>/dev/null | tail -6

echo
echo "--- TRAINING ---"
grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+(s/it|it/s)\]" "$L" 2>/dev/null | tail -1
grep -oE "'loss': [0-9.]+" "$L" 2>/dev/null | tail -5 | tr '\n' ' '
echo
grep -oE "[0-9.]+ samples/s|train_samples_per_second[^,]*" "$L" 2>/dev/null | tail -1

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null

echo
echo "--- DISK ---"
df -h /workspace 2>/dev/null | tail -1 || df -h / | tail -1

echo
echo "--- TROUBLE (should be empty) ---"
grep -iE "Traceback|CUDA out of memory|REFUSING|AccessDenied|No such file or directory" "$L" 2>/dev/null | tail -3

echo
if [ -f /workspace/logs/STAGE_B_DONE ]; then echo "STAGE B COMPLETE"; else echo "running"; fi
