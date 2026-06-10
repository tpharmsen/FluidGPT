#!/bin/bash

nohup konsole --hold -e bash -lc '
ssh -tt snellius "salloc --partition=gpu_mig --time=04:00:00 --gpus-per-node=1 --reservation=terv92681; bash"
' >/dev/null 2>&1 &
