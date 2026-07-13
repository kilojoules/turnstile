"""Provision a vast.ai 96 GB GPU, ship the turnstile package + run_loop.sh,
kick off the single-turn DPO loop detached, and monitor.

Subcommands:
  python vast_run.py launch     # provision + start + monitor (default)
  python vast_run.py monitor    # re-attach to a running instance and tail
  python vast_run.py status
  python vast_run.py destroy
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
PKG = ROOT / "turnstile_pkg.tar.gz"
RUNSH = ROOT / "run_loop.sh"
SESSION = ROOT / "vast_session.json"
KEY = Path(os.path.expanduser("~/.fastai_key")).read_text().strip()
HF_TOKEN = Path(os.path.expanduser("~/.hf_token")).read_text().strip()
SSH_PRIV = os.path.expanduser("~/.ssh/id_ed25519")
SSH_PUB = Path(os.path.expanduser("~/.ssh/id_ed25519.pub")).read_text().strip()

LABEL = "turnstile-singleturn"
IMAGE = "vllm/vllm-openai:latest"
DISK_GB = 250
MAX_PRICE = 1.60
BAD_HOST_IDS = {443829, 9656}  # 443829: no internet; 9656: broken SSH proxy (conn reset)

API = "https://console.vast.ai/api/v0"
H = {"Authorization": f"Bearer {KEY}"}
SSH_OPTS = ["-i", SSH_PRIV, "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]


def search() -> list[dict]:
    q = {
        "verified": {"eq": True}, "rentable": {"eq": True},
        "gpu_ram": {"gte": 90000},          # 96 GB cards
        "num_gpus": {"eq": 1},
        "reliability2": {"gte": 0.97},
        "inet_down": {"gte": 200},
        "disk_space": {"gte": DISK_GB},
        "dph_total": {"lte": MAX_PRICE},
        "cuda_max_good": {"gte": 12.8},
        "order": [["dph_total", "asc"]],
    }
    r = requests.get(f"{API}/bundles/", headers=H, params={"q": json.dumps(q)})
    r.raise_for_status()
    return [o for o in r.json().get("offers", []) if o.get("host_id") not in BAD_HOST_IDS]


ONSTART = """#!/bin/bash
exec > /workspace/onstart.log 2>&1
set -ex
mkdir -p /workspace
# probe HF reachability — bail early if this host is firewalled
if ! curl -fsSL --max-time 15 -o /dev/null https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct; then
  echo HF_UNREACHABLE; touch /workspace/HF_UNREACHABLE; exit 0
fi
pip install -q --no-cache-dir peft jailbreakbench openai 2>&1 | tail -2 || true
echo onstart_done
"""


def launch_offer(ask_id: int) -> dict:
    payload = {
        "client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": LABEL,
        "onstart": ONSTART,
        "env": {"HF_TOKEN": HF_TOKEN, "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
                "OPEN_BUTTON_PORT": "8000"},
        "runtype": "ssh ssh_proxy",
    }
    r = requests.put(f"{API}/asks/{ask_id}/", headers=H, json=payload)
    r.raise_for_status()
    return r.json()


def instances() -> list[dict]:
    r = requests.get(f"{API}/instances/", headers=H)
    r.raise_for_status()
    return r.json().get("instances", [])


def find_ours() -> dict | None:
    for i in instances():
        if i.get("label") == LABEL:
            return i
    return None


def destroy(iid: int):
    r = requests.delete(f"{API}/instances/{iid}/", headers=H)
    print(f"  destroy({iid}) -> {r.status_code} {r.text[:120]}")


def ssh(host, port, cmd, timeout=180):
    r = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def scp_to(host, port, local, remote):
    subprocess.run(["scp", *SSH_OPTS, "-P", str(port), str(local), f"root@{host}:{remote}"],
                   check=True, timeout=180)


def scp_from(host, port, remote, local):
    subprocess.run(["scp", *SSH_OPTS, "-P", str(port), f"root@{host}:{remote}", str(local)],
                   check=True, timeout=180)


def wait_ssh(timeout=1500) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = find_ours()
        if cur and cur.get("ssh_host") and cur.get("ssh_port"):
            rc, out = ssh(cur["ssh_host"], cur["ssh_port"], "echo ok", timeout=20)
            if rc == 0 and "ok" in out:
                print(f"  ssh up at {int(time.time()-t0)}s")
                return cur
        print(f"  waiting for ssh… {int(time.time()-t0)}s")
        time.sleep(20)
    raise TimeoutError("ssh never came up")


def cmd_launch():
    inst = find_ours()
    if inst is None:
        offers = search()
        if not offers:
            print("no offers matched"); sys.exit(1)
        new_id = None
        for o in offers[:8]:
            print(f"trying offer {o['id']} {o['gpu_name']} ${o['dph_total']:.3f} {o.get('geolocation')}")
            try:
                res = launch_offer(o["id"])
            except requests.HTTPError as e:
                print(f"  failed: {(e.response.text if e.response else e)[:140]}"); continue
            if res.get("success"):
                new_id = res["new_contract"]; print(f"launched {new_id}"); break
        if new_id is None:
            print("could not launch"); sys.exit(1)
    else:
        print(f"attaching to existing {inst['id']}")

    inst = wait_ssh()
    host, port = inst["ssh_host"], inst["ssh_port"]
    SESSION.write_text(json.dumps({"id": inst["id"], "host": host, "port": port}))

    # HF reachability gate
    for _ in range(9):
        rc, out = ssh(host, port, "test -f /workspace/HF_UNREACHABLE && echo UNREACH; test -f /workspace/onstart.log && tail -1 /workspace/onstart.log")
        if "UNREACH" in out:
            print("HF unreachable on this host; destroying"); destroy(inst["id"]); SESSION.unlink(missing_ok=True); sys.exit(2)
        if "onstart_done" in out:
            break
        time.sleep(10)

    print("shipping package + run script…")
    ssh(host, port, "mkdir -p /workspace")
    scp_to(host, port, PKG, "/workspace/turnstile_pkg.tar.gz")
    scp_to(host, port, RUNSH, "/workspace/run_loop.sh")

    print("kicking off run_loop.sh detached…")
    ssh(host, port, f"export HF_TOKEN={HF_TOKEN}; cd /workspace && nohup bash run_loop.sh >/dev/null 2>&1 & echo started")
    print("started. monitor with: python vast_run.py monitor")
    cmd_monitor(host, port, inst["id"])


def cmd_monitor(host=None, port=None, iid=None):
    if host is None:
        s = json.loads(SESSION.read_text())
        host, port, iid = s["host"], s["port"], s["id"]
    print(f"monitoring root@{host}:{port} (instance {iid})")
    t0 = time.time()
    last = ""
    while True:
        rc, out = ssh(host, port,
                      "test -f /workspace/DONE && echo __DONE__; "
                      "tail -n 4 /workspace/run.log 2>/dev/null; "
                      "echo ---METRICS---; "
                      "cat /workspace/turnstile_src/experiments/singleturn_hardened_v1/metrics.jsonl 2>/dev/null | tail -6",
                      timeout=60)
        el = int(time.time() - t0)
        if out != last:
            print(f"\n[{el}s] " + out.strip())
            last = out
        else:
            print(".", end="", flush=True)
        if "__DONE__" in out:
            print(f"\n=== DONE at {el}s ===")
            break
        if el > 14400:  # 4 h hard cap
            print("\n=== monitor 4h cap reached; leaving instance up ===")
            return
        time.sleep(60)

    # pull results
    for remote in ("/workspace/run.log", "/workspace/loop.log", "/workspace/judge.log",
                   "/workspace/victim.log",
                   "/workspace/turnstile_src/experiments/singleturn_hardened_v1/metrics.jsonl"):
        try:
            scp_from(host, port, remote, ROOT / Path(remote).name)
        except Exception as e:
            print(f"  scp {remote} failed: {e}")
    print("results pulled. NOT destroying — inspect, then `python vast_run.py destroy`.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "launch"
    if cmd == "launch":
        cmd_launch()
    elif cmd == "monitor":
        cmd_monitor()
    elif cmd == "status":
        print(json.dumps(find_ours(), indent=2, default=str))
    elif cmd == "destroy":
        inst = find_ours()
        if inst:
            destroy(inst["id"]); SESSION.unlink(missing_ok=True)
        else:
            print("no instance")
    else:
        print(f"unknown: {cmd}")


if __name__ == "__main__":
    main()
