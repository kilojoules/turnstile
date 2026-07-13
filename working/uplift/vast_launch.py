"""End-to-end launcher: provision a vast.ai A100, start vLLM serving
Qwen2.5-72B-Instruct-AWQ, scp the judge script + goals, run, pull results,
destroy.

Subcommands:
  python vast_launch.py launch    # full flow (default)
  python vast_launch.py status    # show our instance, if any
  python vast_launch.py destroy   # tear down our instance

Re-runnable: identifies our instance by `INSTANCE_LABEL`, attaches if it
already exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
# Stage selection via env var so we don't fork the launcher
STAGE = os.environ.get("STAGE", "b")
if STAGE == "a":
    INPUT_FILE = ROOT / "goals.json"
    JUDGE = ROOT / "run_judge.py"
    SCORES = ROOT / "stage_a_scores.jsonl"
    REMOTE_INPUT = "/workspace/goals.json"
    REMOTE_OUT = "/workspace/stage_a_scores.jsonl"
    JUDGE_BASENAME = "run_judge.py"
else:
    INPUT_FILE = ROOT / "stage_b_candidates.jsonl"
    JUDGE = ROOT / "run_judge_b.py"
    SCORES = ROOT / "stage_b_scores.jsonl"
    REMOTE_INPUT = "/workspace/stage_b_candidates.jsonl"
    REMOTE_OUT = "/workspace/stage_b_scores.jsonl"
    JUDGE_BASENAME = "run_judge_b.py"
SESSION = ROOT / "vast_session.json"
KEY = Path(os.path.expanduser("~/.fastai_key")).read_text().strip()
HF_TOKEN = Path(os.path.expanduser("~/.hf_token")).read_text().strip()
SSH_PUB = Path(os.path.expanduser("~/.ssh/id_ed25519.pub")).read_text().strip()
SSH_PRIV = os.path.expanduser("~/.ssh/id_ed25519")

INSTANCE_LABEL = f"turnstile-stage-{STAGE}"
MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
IMAGE = "vllm/vllm-openai:latest"
DISK_GB = 100
MAX_PRICE = 1.50  # $/hr ceiling

API = "https://console.vast.ai/api/v0"
H = {"Authorization": f"Bearer {KEY}"}


# -------------------- vast helpers --------------------

BAD_HOST_IDS = {443829}  # 192.3.91.246: no outbound internet, broken sshd key propagation
BAD_OFFER_FILE = ROOT / "vast_bad_offers.json"


def _load_bad_offers() -> set[int]:
    if BAD_OFFER_FILE.exists():
        return set(json.loads(BAD_OFFER_FILE.read_text()))
    return set()


def _record_bad_offer(offer_id: int):
    bad = _load_bad_offers()
    bad.add(offer_id)
    BAD_OFFER_FILE.write_text(json.dumps(sorted(bad)))


def vast_search() -> list[dict]:
    q = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "gpu_ram": {"gte": 80000},
        "num_gpus": {"eq": 1},
        "reliability2": {"gte": 0.97},
        "inet_down": {"gte": 200},
        "disk_space": {"gte": DISK_GB},
        "dph_total": {"lte": MAX_PRICE},
        "cuda_max_good": {"gte": 12.8},   # vllm/vllm-openai:latest needs CUDA >= 12.8
        "order": [["dph_total", "asc"]],
    }
    r = requests.get(f"{API}/bundles/", headers=H, params={"q": json.dumps(q)})
    r.raise_for_status()
    bad_offers = _load_bad_offers()
    return [o for o in r.json().get("offers", [])
            if o.get("host_id") not in BAD_HOST_IDS and o.get("id") not in bad_offers]


ONSTART = """#!/bin/bash
exec > /workspace/onstart.log 2>&1
set -ex
mkdir -p /workspace
echo "onstart begin at $(date)"
echo "probing HF reachability"
if ! curl -fsSL --max-time 15 -o /dev/null https://huggingface.co/api/models/{model}; then
    echo "HF UNREACHABLE — aborting vllm start"
    touch /workspace/HF_UNREACHABLE
    exit 0
fi
echo "HF reachable"
nohup python3 -m vllm.entrypoints.openai.api_server \\
    --model {model} \\
    --port 8000 \\
    --max-model-len 8192 \\
    --gpu-memory-utilization 0.92 \\
    --quantization awq_marlin \\
    --download-dir /workspace/hf_cache \\
    > /workspace/vllm.log 2>&1 &
echo "vllm launched at $(date)"
"""


def vast_launch(ask_id: int) -> dict:
    payload = {
        "client_id": "me",
        "image": IMAGE,
        "disk": DISK_GB,
        "label": INSTANCE_LABEL,
        "onstart": ONSTART.format(model=MODEL),
        "env": {
            "HF_TOKEN": HF_TOKEN,
            "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
            "OPEN_BUTTON_PORT": "8000",
        },
        "runtype": "ssh ssh_proxy",
    }
    r = requests.put(f"{API}/asks/{ask_id}/", headers=H, json=payload)
    r.raise_for_status()
    return r.json()


def vast_instances() -> list[dict]:
    r = requests.get(f"{API}/instances/", headers=H)
    r.raise_for_status()
    return r.json().get("instances", [])


def vast_destroy(instance_id: int):
    r = requests.delete(f"{API}/instances/{instance_id}/", headers=H)
    print(f"  destroy({instance_id}) → {r.status_code} {r.text[:200]}")


def vast_attach_key():
    """Vast stores SSH keys on the user record. Just verify ours is present."""
    r = requests.get(f"{API}/users/current", headers=H)
    r.raise_for_status()
    keys = r.json().get("ssh_key", "") or ""
    fingerprint = SSH_PUB.split()[1]
    if fingerprint in keys:
        print("  ssh key already registered")
    else:
        print("  WARNING: ssh key not in user record; SSH may fail. "
              "Add via web UI: https://cloud.vast.ai/account/")


# -------------------- ssh / scp --------------------

SSH_OPTS = ["-i", SSH_PRIV,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=10"]


def ssh_cmd(host: str, port: int, cmd: str, timeout: int = 120) -> tuple[int, str]:
    full = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


def scp_to(host: str, port: int, local: Path, remote: str):
    full = ["scp", *SSH_OPTS, "-P", str(port), str(local), f"root@{host}:{remote}"]
    subprocess.run(full, check=True, timeout=120)


def scp_from(host: str, port: int, remote: str, local: Path):
    full = ["scp", *SSH_OPTS, "-P", str(port), f"root@{host}:{remote}", str(local)]
    subprocess.run(full, check=True, timeout=120)


# -------------------- main flow --------------------

def find_ours() -> dict | None:
    for inst in vast_instances():
        if inst.get("label") == INSTANCE_LABEL:
            return inst
    return None


def wait_ssh(timeout: int = 900) -> dict:
    """Wait until vast assigns ssh host/port AND the sshd in the container actually accepts a connection."""
    t0 = time.time()
    cur = None
    while time.time() - t0 < timeout:
        cur = find_ours()
        if cur and cur.get("ssh_host") and cur.get("ssh_port"):
            host, port = cur["ssh_host"], cur["ssh_port"]
            rc, out = ssh_cmd(host, port, "echo ok", timeout=20)
            if rc == 0 and "ok" in out:
                print(f"  ssh up at {int(time.time()-t0)}s: root@{host}:{port}")
                return cur
            tag = (out or "").strip().splitlines()[-1] if out else ""
            print(f"  ssh attempt at {int(time.time()-t0)}s rc={rc}: {tag[:120]}")
        else:
            status = cur.get("actual_status", "?") if cur else "?"
            print(f"  waiting for ssh meta… status={status}  age={int(time.time()-t0)}s")
        time.sleep(20)
    raise TimeoutError("ssh never came up")


def wait_vllm_ready(host: str, port: int, timeout: int = 1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rc, out = ssh_cmd(host, port,
                          "curl -fs http://localhost:8000/v1/models 2>&1 | head -c 200")
        elapsed = int(time.time() - t0)
        if rc == 0 and ("data" in out or "object" in out):
            print(f"  vllm ready at {elapsed}s")
            return
        # Show tail of vllm log for visibility
        rc2, tail = ssh_cmd(host, port, "tail -n 2 /workspace/vllm.log 2>/dev/null")
        print(f"  [{elapsed:4d}s] vllm not ready. log tail: {tail.strip()[:200]}")
        time.sleep(30)
    raise TimeoutError("vllm never came up")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"

    if cmd == "destroy":
        inst = find_ours()
        if inst:
            vast_destroy(inst["id"])
            if SESSION.exists():
                SESSION.unlink()
        else:
            print("no instance to destroy")
        return

    if cmd == "status":
        inst = find_ours()
        print(json.dumps(inst, indent=2, default=str) if inst else "no instance")
        return

    vast_attach_key()
    inst = find_ours()
    if inst is None:
        offers = vast_search()
        if not offers:
            print("NO offers matched", file=sys.stderr); sys.exit(1)
        new_id = None
        for chosen in offers[:8]:
            print(f"trying offer: id={chosen['id']}  gpu={chosen['gpu_name']}  "
                  f"dph=${chosen['dph_total']:.3f}  loc={chosen.get('geolocation')}")
            try:
                result = vast_launch(chosen["id"])
            except requests.exceptions.HTTPError as e:
                msg = e.response.text[:200] if e.response is not None else str(e)
                print(f"  failed: {msg}")
                continue
            if result.get("success"):
                new_id = result["new_contract"]
                print(f"launched id={new_id}")
                break
            print(f"  no success: {result}")
        if new_id is None:
            print("could not launch any offer", file=sys.stderr); sys.exit(1)
    else:
        print(f"attaching to existing instance id={inst['id']}")

    inst = wait_ssh(timeout=900)
    SESSION.write_text(json.dumps(inst, indent=2, default=str))
    host, port = inst["ssh_host"], inst["ssh_port"]

    # Sanity-check: did the onstart's HF reachability probe pass?
    print("checking HF reachability marker (waiting up to 90s for onstart)…")
    for _ in range(9):
        rc1, _ = ssh_cmd(host, port, "test -f /workspace/HF_UNREACHABLE && echo UNREACH || true")
        rc2, vllog = ssh_cmd(host, port, "test -f /workspace/vllm.log && echo VLLM || true")
        if "UNREACH" in vllog or "UNREACH" in (vllog or ""):
            pass
        rc, marker = ssh_cmd(host, port,
                             "test -f /workspace/HF_UNREACHABLE && echo UNREACH; "
                             "test -f /workspace/vllm.log && echo VLLM_OK")
        if "UNREACH" in marker:
            print("  HF UNREACHABLE on this host — destroying and aborting")
            vast_destroy(inst["id"])
            if SESSION.exists(): SESSION.unlink()
            sys.exit(2)
        if "VLLM_OK" in marker:
            print("  HF reachable, vllm launching")
            break
        time.sleep(10)

    print(f"scp input + judge script (stage {STAGE})…")
    ssh_cmd(host, port, "mkdir -p /workspace")
    scp_to(host, port, INPUT_FILE, REMOTE_INPUT)
    scp_to(host, port, JUDGE, f"/workspace/{JUDGE_BASENAME}")
    # Resume support: ship existing partial scores up so the judge can skip them
    if SCORES.exists() and SCORES.stat().st_size > 0:
        print(f"  shipping partial scores ({SCORES.stat().st_size} bytes) for resume…")
        scp_to(host, port, SCORES, REMOTE_OUT)

    print("waiting for vllm to be ready…")
    wait_vllm_ready(host, port, timeout=1800)

    print("running judge…")
    rc, out = ssh_cmd(host, port,
                      f"cd /workspace && python3 {JUDGE_BASENAME} 2>&1 | tee judge.log",
                      timeout=2400)
    print(out)
    print(f"judge rc={rc}")

    print("pulling results…")
    try:
        scp_from(host, port, REMOTE_OUT, SCORES)
        print(f"wrote {SCORES}")
    except Exception as e:
        print(f"scp results failed: {e}")
    for remote in ("/workspace/judge.log", "/workspace/vllm.log",
                   "/workspace/onstart.log"):
        try:
            scp_from(host, port, remote, ROOT / Path(remote).name)
        except Exception:
            pass

    print("destroying instance…")
    vast_destroy(inst["id"])
    if SESSION.exists():
        SESSION.unlink()


if __name__ == "__main__":
    main()
