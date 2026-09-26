#!/bin/bash
# Read-only facts about your CLAIX account and the node, for setting up
# AQI-HIL-SERL. Changes nothing, prints no secrets. Run on a login node:
#
#   bash probe.sh 2>&1 | tee probe.txt
#
# then paste probe.txt back. On login23-g-1 it also reports the GPUs.

sec() { printf '\n===== %s =====\n' "$*"; }

sec "node"
hostname; cat /etc/redhat-release 2>/dev/null; uname -r
echo "user=$(id -un)  groups=$(id -Gn | tr ' ' ',')"
echo "HOME=$HOME  WORK=${WORK:-unset}  HPCWORK=${HPCWORK:-unset}"

sec "storage quota (r_quota)"
r_quota 2>/dev/null || echo "r_quota not available"

sec "computing time (r_wlm_usage)"
r_wlm_usage 2>/dev/null || echo "r_wlm_usage not available"

sec "Slurm accounts / projects you can charge"
sacctmgr -nP show assoc user="$(id -un)" format=account,partition,qos 2>/dev/null \
    | sort -u || echo "sacctmgr not available"

sec "GPU partitions"
sinfo -s -p c23g,devel 2>/dev/null || echo "sinfo not available"

sec "Python"
python3 --version 2>&1
command -v uv >/dev/null && uv --version
module -t avail Python 2>&1 | grep -i python | tail -5

sec "internet from this node"
for u in https://pypi.org/simple/ https://github.com https://files.pythonhosted.org \
         https://api.wandb.ai; do
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u")
    echo "$u -> ${code}"
done

sec "graphics / USB libraries"
ldconfig -p 2>/dev/null | grep -E 'libEGL\.so|libEGL_nvidia|libOSMesa|libGL\.so\.1|libhidapi|libudev\.so' \
    | awk '{print $1}' | sort -u
ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null

sec "GPUs on this node"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader | head -1
else
    echo "no nvidia-smi (normal on login23-1/-2/-4; use login23-g-1 for GPUs)"
fi

sec "tools"
git --version; tmux -V 2>/dev/null; rsync --version 2>/dev/null | head -1
