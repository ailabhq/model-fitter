#!/usr/bin/env python3
"""Rent, inspect and terminate the one RunPod GPU pod that runs the sensitivity probes.

Usage: runpod_pod.py balance | create [--community] [--gpu GPU_TYPE_ID] [--disk GB] | ssh | terminate

The API key is read from RunPod's own login store (~/.runpod/config.toml, written by `runpodctl config`)
or RUNPOD_API_KEY, and is never printed. The pod id is kept in ./.pod-id (untracked).
"""
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

IMAGE = "runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404"
GPU = "NVIDIA H100 80GB HBM3"
POD_FILE = Path(".pod-id")


def key() -> str:
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    return tomllib.loads((Path.home() / ".runpod" / "config.toml").read_text())["default"]["api_key"]


def gql(query: str) -> dict:
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": query}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key()}", "User-Agent": "model-fitter"})
    out = json.load(urllib.request.urlopen(req))
    if out.get("errors"):
        sys.exit(json.dumps(out["errors"]))
    return out["data"]


def main():
    cmd = sys.argv[1]
    if cmd == "balance":
        print(json.dumps(gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]))
    elif cmd == "create":
        cloud = "COMMUNITY" if "--community" in sys.argv else "SECURE"
        gpu = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else GPU
        disk = int(sys.argv[sys.argv.index("--disk") + 1]) if "--disk" in sys.argv else 300
        pub = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
        d = gql(f"""mutation {{ podFindAndDeployOnDemand(input: {{
            cloudType: {cloud}, gpuCount: 1, gpuTypeId: "{gpu}", name: "sarto-sensitivity", imageName: "{IMAGE}",
            containerDiskInGb: {disk}, volumeInGb: 0, ports: "22/tcp", supportPublicIp: true, startSsh: true,
            env: [{{ key: "PUBLIC_KEY", value: {json.dumps(pub)} }}] }}) {{ id costPerHr machine {{ gpuDisplayName }} }} }}""")
        pod = d["podFindAndDeployOnDemand"]
        POD_FILE.write_text(pod["id"])
        print(json.dumps({"created": True, "costPerHr": pod["costPerHr"], "gpu": pod["machine"]["gpuDisplayName"]}))
    elif cmd == "ssh":
        pid = POD_FILE.read_text().strip()
        p = gql(f'{{ pod(input: {{podId: "{pid}"}}) {{ desiredStatus runtime {{ uptimeInSeconds ports {{ ip isIpPublic privatePort publicPort }} }} }} }}')["pod"]
        ports = [x for x in ((p.get("runtime") or {}).get("ports") or []) if x["privatePort"] == 22 and x["isIpPublic"]]
        print(f"root@{ports[0]['ip']} -p {ports[0]['publicPort']}" if ports else json.dumps({"status": p["desiredStatus"], "ready": False}))
    elif cmd == "terminate":
        pid = POD_FILE.read_text().strip()
        gql(f'mutation {{ podTerminate(input: {{podId: "{pid}"}}) }}')
        POD_FILE.unlink()
        print(json.dumps({"terminated": True}))


if __name__ == "__main__":
    main()
