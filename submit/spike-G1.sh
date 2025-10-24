#!/usr/bin/env bash

set -a
source .env
set +a

MULTIPLIER=${1:-1}

CPU_CORE_REQUEST=$((4 * MULTIPLIER))
CPU_CORE_LIMIT=$((25 * MULTIPLIER))
CPU_MEMORY_REQUEST=$((32 * MULTIPLIER))G
CPU_MEMORY_LIMIT=$((245 * MULTIPLIER))G
GPU_DEVICES_REQUEST=$((1 * MULTIPLIER))

echo "GPU devices request: $GPU_DEVICES_REQUEST"

runai workspace submit fluidgpt-testsubmit-cli \
--image "harbor.spike.tue.nl/fluidgpt/fluidgpt-fm-ssh:latest" \
--project "fluidgpt" \
--cpu-core-limit $CPU_CORE_LIMIT \
--cpu-core-request $CPU_CORE_REQUEST \
--cpu-memory-limit $CPU_MEMORY_LIMIT \
--cpu-memory-request $CPU_MEMORY_REQUEST \
--gpu-devices-request $GPU_DEVICES_REQUEST \
--backoff-limit 1   \
--new-pvc "claimname=shm-ephemeral,storageclass=exascaler-ephemeral,size=512G,path=/dev/shm,accessmode-rwm,ephemeral" \
--existing-pvc "claimname=fluidgpt,path=/data" \
--git-sync "name=fluidgpt,repository=https://github.com/tpharmsen/FluidGPT,path=/code/,rev=main" \
--environment WANDB_API_KEY="$VAR3" \
--command -- torchrun --nproc-per-node=$GPU_DEVICES_REQUEST src/train.py --trainer FM --CB spike-high --CD spike-gauss --CM fm-test-B --CT fm-b200-doublegauss6 --out SpikeCLI-test