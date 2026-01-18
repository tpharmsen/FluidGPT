#!/usr/bin/env bash

#set -a
#source .env
#set +a

# Check required variable(s)
#: "${WANDB_API_KEY:?WANDB_API_KEY not set in any ENV file}"

#runai login user -u="$VAR1" -p="$VAR2" --quiet
runai workspace submit fgpt-fm-semifinal \
--image "harbor.spike.tue.nl/fluidgpt/fluidgpt-fm-ssh:latest" \
--project "fluidgpt" \
--user-group-source "fromTheImage" \
--cpu-core-limit 25 \
--cpu-core-request 4 \
--cpu-memory-limit  245G \
--cpu-memory-request 32G \
--gpu-devices-request 1 \
--backoff-limit 1 \
--new-pvc "claimname=shm-ephemeral,storageclass=exascaler-ephemeral,size=512G,path=/dev/shm,accessmode-rwm,ephemeral" \
--existing-pvc "claimname=fluidgpt,path=/data" \
--git-sync "name=fluidgpt,repository=https://github.com/tpharmsen/FluidGPT,path=/code/,rev=main" \
--environment WANDB_API_KEY="$VAR3" \
--command -- torchrun --nproc-per-node=8 src/train.py --trainer FM --CB spike-high --CD spike-preprocAll --CM fm-post-d3 --CT fm-semifinal --out fm-post-d3-run