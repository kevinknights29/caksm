#!/bin/bash
# Record what the machine actually is, before any kernel runs on it.
#
# Nothing here needs the project built or a GPU kernel launched: it reads
# nvidia-smi, the fabric and the scheduler. That bounds what it can answer.
# Identity, capacity and topology come from here; device geometry does not,
# because nvidia-smi does not report it, and no roof or reduction cost does,
# because those are measured. The closing message says which field comes from
# where. Two of the checks decide whether the study can proceed at all.
# A MIG slice partitions the L2 that the vertical coordinate is measured
# against, so 6 MiB is then the wrong number, and a co-tenant process shares
# that L2 with no way to account for it. The inter-node fabric, Ethernet or
# InfiniBand, moves the node reduction rung by an order of magnitude, and that
# rung is the right-hand end of the swept horizontal axis.
#
# Run it once per node. The script probes the node it runs on, and names its
# output after that host, so a two-node allocation needs one task per node:
#
#   salloc -N 2 -n 20 -p compute -t 01:00:00 --nodelist=synge-n01,synge-n02
#   srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=2 \
#       scripts/regime/gpu_probe.sh
#
# A plain `bash scripts/regime/gpu_probe.sh` inside the same allocation runs on
# the first node only, which is the failure mode this note exists to prevent.
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

echo "### Devices: name, compute capability, memory and link width"
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
for t in nvcc mpicc mpicxx mpirun nvidia-smi cmake; do
    if command -v "$t" >/dev/null 2>&1; then
        echo "  $t: $(command -v "$t")"
        case "$t" in
            nvcc)   nvcc --version | tail -2 | sed 's/^/      /' ;;
            mpicc)  mpicc --version 2>/dev/null | head -1 | sed 's/^/      /' ;;
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
echo "include/gpu_machine.hpp fields this probe gives you:"
echo "    name, device_match       nvidia-smi -L"
echo "    device_memory_bytes      memory.total"
echo "    gpu_count                GPU lines listed for this host"
echo "    node_count               the nodes: line above"
echo "    interconnect             topo matrix cell (SYS, NV#, PIX, ...)"
echo
echo "nvidia-smi does not report device geometry. gpu-stream"
echo "prints SMs and L2 in its header. All four are cudaGetDeviceProperties:"
echo "    sm_count                 multiProcessorCount"
echo "    l2_bytes                 l2CacheSize"
echo "    shared_mem_bytes_per_sm  sharedMemPerMultiprocessor"
echo "    warps_per_sm             maxThreadsPerMultiProcessor / 32"
echo
echo "Every roof and every t_reduce entry is measured, never transcribed:"
echo "    scripts/regime/calibrate_gpu.sh, then calibrate_gpu_p2p.sh"
echo
echo "Stop if MIG mode is not [N/A], MPS is running, or a listed process is not"
echo "yours: the L2 that R_v divides by is then partitioned or shared."
echo "==============================================================================="
