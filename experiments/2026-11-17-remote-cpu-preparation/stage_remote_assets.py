#!/usr/bin/env python3
"""Run only on the authorized remote CPU; keep durable assets on its data volume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

IMAGE = 'vllm/vllm-openai@sha256:bf0486e51765766a069f6eb26a63a06748859a568b24200acd3aa4edb58f3d35'
CONFIG = 'sha256:aef4ebc906574fa2e6e52079937b7d3a8735e218f6353b58667b215e4257a34b'
TAG = 'edmir/vllm:0.20.1-locked'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def run(command, log, deadline, timeout=5400):
    remaining = min(timeout, deadline - time.time())
    if remaining <= 0:
        raise TimeoutError('remote preparation deadline reached')
    with log.open('w') as f:
        p = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = p.wait(timeout=remaining)
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise
    if result:
        raise RuntimeError(f'{Path(command[0]).name} exited {result}; see {log.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--code', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--deadline', type=float, required=True)
    a = ap.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('remote Linux amd64 only')
    if list(Path('/dev').glob('nvidia[0-9]*')):
        raise RuntimeError('GPU device present: refusing this CPU-only task')
    guard = json.loads((a.root / 'volume-identity.json').read_text())
    if guard['volume_id'] != '5c647cb1-fed1-48ed-b7bf-52fd04c329b0' or not guard['mount_verified']:
        raise RuntimeError('wrong/unverified persistent volume')
    if not time.time() < a.deadline <= time.time() + 4 * 3600:
        raise ValueError('invalid preparation deadline')
    a.root = a.root.resolve()
    records = a.root / 'records'
    records.mkdir(exist_ok=True)
    state = dict(started_utc=stamp(), status='running', GPU_used=False,
                 root=str(a.root), stages={}, deadline_epoch=a.deadline)
    save(records / 'preparation-status.json', state)
    tools = a.code / 'experiments/2026-11-16-offline-assets'
    manifest = a.code / 'experiments/2026-11-16-next-window/model-manifest.json'

    def model():
        run([str(a.python), str(tools / 'download_model.py'), '--manifest', str(manifest),
             '--root', str(a.root), '--records', str(records)], records / 'model-download.log', a.deadline)
        return dict(status='verified', receipt='model-download-status.json', finished_utc=stamp())

    def image():
        run(['docker', 'pull', '--platform', 'linux/amd64', IMAGE], records / 'image-pull.log', a.deadline)
        inspection = json.loads(subprocess.check_output(['docker', 'image', 'inspect', IMAGE], timeout=30))[0]
        save(records / 'image-inspect.json', inspection)
        # Docker 29/containerd identifies images by the manifest; the legacy
        # graphdriver uses the config digest. Verify repository identity too.
        if inspection['Id'] not in (CONFIG, IMAGE.split('@')[1]) or inspection['Architecture'] != 'amd64' or inspection['Os'] != 'linux':
            raise ValueError('wrong image identity/platform')
        if inspection['Id'] == IMAGE.split('@')[1] and inspection.get('Descriptor', {}).get('digest') != IMAGE.split('@')[1]:
            raise ValueError('containerd image descriptor differs')
        if IMAGE not in inspection.get('RepoDigests', []):
            raise ValueError('pinned repository digest absent')
        run(['docker', 'tag', IMAGE, TAG], records / 'image-tag.log', a.deadline, 30)
        archive = a.root / 'vllm-0.20.1-amd64.docker.tar'
        partial = archive.with_suffix('.partial')
        run(['docker', 'save', '--output', str(partial), TAG], records / 'image-save.log', a.deadline, 1800)
        partial.replace(archive)
        receipt = dict(status='archived', archive=str(archive), bytes=archive.stat().st_size,
                       sha256=sha(archive), manifest_digest=IMAGE.split('@')[1], config_digest=CONFIG,
                       platform='linux/amd64', tag=TAG, GPU_used=False,
                       reload_verified=False, finished_utc=stamp())
        save(records / 'image-archive-receipt.json', receipt)
        return receipt

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {'model': pool.submit(model), 'image': pool.submit(image)}
        while jobs:
            for name, job in list(jobs.items()):
                if not job.done():
                    continue
                try:
                    state['stages'][name] = job.result()
                except BaseException as exc:
                    state['stages'][name] = dict(status='failed', error=repr(exc), finished_utc=stamp())
                del jobs[name]
                save(records / 'preparation-status.json', state)
            if jobs:
                time.sleep(2)
    state['status'] = 'assets_downloaded' if all(s['status'] in ('verified', 'archived') for s in state['stages'].values()) else 'failed'
    state['finished_utc'] = stamp()
    save(records / 'preparation-status.json', state)
    if state['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
