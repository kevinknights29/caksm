#!/bin/bash
# Phase 0 fact collection: the machine questions that must be answered before any kernel.
#
# Establishes exactly what hardware synge is, and whether the CPU study's topology objection
# applies here. Neither needs the project built and neither needs a GPU kernel: this is
# nvidia-smi, the fabric and the scheduler, recorded so the constants in
# include/gpu_machine.hpp have a provenance rather than a datasheet.
#
# Two of these checks are not formalities:
#
#   MIG / MPS / co-tenancy.  A MIG slice reports a partitioned L2, real unlike a VM's invented one,
#   but it must then be modeled as the partition rather than the full 6 MiB. A co-tenant process on
#   the same device is worse: it shares the L2 the vertical coordinate is measured against,
#   with no way to account for it.
#
#   The inter-node fabric.  Ethernet or InfiniBand changes the NODE rung's latency by an order
#   of magnitude, and that rung is the right-hand end of the swept horizontal axis.
#
# Run this under the batch scheduler.
#
# Usage:
#   sbatch --nodes=2 --gpus-per-node=2 scripts/regime/gpu_probe.sh
set -uo pipefail

WORK_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$WORK_DIR/data/regime"
mkdir -p "$DATA_DIR"
OUT="$DATA_DIR/gpu_probe_$(uname -n).txt"

exec > >(tee "$OUT") 2>&1

echo "==============================================================================="
echo " GPU probe   host=$(uname -n)   date=$(date -Is)"
echo "==============================================================================="
echo

echo "### Host and allocation"
echo "  kernel   : $(uname -srm)"
echo "  slurm job: ${SLURM_JOB_ID:-<none: not under the scheduler>}"
echo "  nodes    : ${SLURM_JOB_NODELIST:-<none>}"
echo "  gpus     : ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-<unset>}}"
echo "  exclusive: ${SLURM_JOB_OVERSUBSCRIBE:-<unknown>}"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|Socket|NUMA node\(s\)|NUMA node[0-9]' | sed 's/^/  cpu    : /'
fi
echo

echo "### Devices: confirm sm_count, l2_bytes, memory and FP64 class"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  nvidia-smi not found. Nothing below can be trusted; load the CUDA module first."
else
    nvidia-smi -L | sed 's/^/  /'
    echo
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max \
               --format=csv | sed 's/^/  /'
    echo
    echo "  Persistence / compute mode (a non-default mode changes who may share the device):"
    nvidia-smi --query-gpu=index,persistence_mode,compute_mode --format=csv | sed 's/^/  /'
fi
echo

echo "### Topology: which link the DEVICE_P2P rung actually crosses"
echo "  'SYS' = PCIe + cross-socket UPI (a genuine cross-socket hop, no NVLink)."
echo "  'NV#' = NVLink, which would make the intra-node rung nearly free and cost the study"
echo "          its cheapest horizontal probe. Record whichever it is."
nvidia-smi topo -m 2>/dev/null | sed 's/^/  /' || echo "  (topo unavailable)"
echo

echo "### MIG, MPS and co-tenancy: the topology objection, applied to a GPU"
echo "  MIG mode (enabled means L2 is partitioned and gpu_machine.hpp's 6 MiB is wrong):"
nvidia-smi --query-gpu=index,mig.mode.current --format=csv 2>/dev/null | sed 's/^/    /' \
    || echo "    (unavailable)"
echo "  MIG instances, if any:"
nvidia-smi -L 2>/dev/null | grep -i mig | sed 's/^/    /' || echo "    none"
echo
echo "  MPS control daemon (a running daemon means other jobs may share your SMs):"
pgrep -a nvidia-cuda-mps-control 2>/dev/null | sed 's/^/    /' || echo "    not running"
echo
echo "  Processes currently resident on each device (any entry not yours is a confound:"
echo "  a co-tenant shares the L2 that R_v is measured against, unaccountably):"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null \
    | sed 's/^/    /' || echo "    (unavailable)"
echo

echo "### Inter-node fabric: sets the NODE rung by an order of magnitude"
if command -v ibstat >/dev/null 2>&1; then
    echo "  InfiniBand present:"
    ibstat 2>/dev/null | grep -E 'CA |State:|Physical state:|Rate:|Link layer:' | sed 's/^/    /'
else
    echo "  ibstat not found: no InfiniBand tooling on this node."
fi
echo "  Network interfaces and link speeds:"
if command -v ip >/dev/null 2>&1; then
    ip -br link 2>/dev/null | sed 's/^/    /'
    for dev in $(ls /sys/class/net 2>/dev/null); do
        speed=$(cat "/sys/class/net/$dev/speed" 2>/dev/null || echo "?")
        [[ "$speed" != "?" ]] && echo "    $dev: ${speed} Mb/s"
    done
else
    echo "    (ip not available)"
fi
echo "  GPUDirect RDMA (nvidia_peermem / nv_peer_mem loaded means NCCL can bypass the host):"
lsmod 2>/dev/null | grep -E 'nvidia_peermem|nv_peer_mem' | sed 's/^/    /' || echo "    not loaded"
echo

echo "### Toolchain"
for t in nvcc mpicxx nvidia-smi cmake; do
    if command -v "$t" >/dev/null 2>&1; then
        echo "  $t: $(command -v "$t")"
        case "$t" in
            nvcc)   nvcc --version | tail -2 | sed 's/^/      /' ;;
            mpicxx) mpicxx --version 2>/dev/null | head -1 | sed 's/^/      /' ;;
        esac
    else
        echo "  $t: not found"
    fi
done
echo "  NCCL: ${NCCL_HOME:-${NCCL_ROOT:-<NCCL_HOME/NCCL_ROOT unset>}}"
find /usr /opt "${NCCL_HOME:-/nonexistent}" -name 'nccl.h' 2>/dev/null | head -3 | sed 's/^/    /'
echo

echo "==============================================================================="
echo "Wrote $OUT"
echo
echo "What to do with it:"
echo "  1. Transcribe sm_count, l2_bytes and memory into the preset in"
echo "     include/gpu_machine.hpp and confirm they match the datasheet priors."
echo "  2. If MIG is enabled or a co-tenant process appears, stop. Both coordinates are"
echo "     functions of real cache geometry, so a partitioned or shared L2 must be modeled"
echo "     as such rather than assumed away. This is the same objection the README raises"
echo "     against cloud VMs, and it does not stop applying because the partition is real."
echo "  3. Record the fabric identity. It sets the NODE rung, which is the right-hand end of"
echo "     the swept horizontal axis."
echo "==============================================================================="
