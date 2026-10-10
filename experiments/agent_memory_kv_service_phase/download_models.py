"""Download authorized public ModelScope weights in a CPU-only isolated env."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time


def download(repo, destination, logs):
    from modelscope import snapshot_download
    from modelscope.hub.api import HubApi
    api = HubApi();revision = api.get_valid_revision(repo)
    remote = api.get_model_files(repo, revision=revision)
    path = destination / repo.split('/')[-1]
    snapshot_download(repo, revision=revision, local_dir=str(path), max_workers=4,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.json', 'LICENSE', 'README.md'])
    index = path / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    expected = {f['Path']: f.get('Size') for f in remote}
    hashes = {}
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', *files]:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise RuntimeError(f'Missing model artifact: {file}')
        if expected.get(name) is not None and file.stat().st_size != expected[name]:
            raise RuntimeError(f'Incomplete model artifact: {file}')
        h = hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):h.update(block)
        hashes[name] = h.hexdigest()
    info = {'provider': 'modelscope', 'repo': repo, 'resolved_revision': revision,
            'branch_revision_may_be_mutable': revision == 'master', 'path': str(path.resolve()),
            'files_sha256': hashes, 'remote_file_listing': remote, 'finish_epoch': time.time(),
            'weights_complete': True, 'GPU_inference_validated': False}
    (logs / f'{repo.split("/")[-1]}-weights.lock.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({'repo': repo, 'path': str(path), 'weights_complete': True}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True);p.add_argument('--models', nargs='+', required=True)
    a = p.parse_args();os.environ['CUDA_VISIBLE_DEVICES'] = ''
    a.logs.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(download, repo, a.destination, a.logs) for repo in a.models]
        for job in jobs:job.result()
