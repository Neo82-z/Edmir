"""Freeze the local review, then verify and reproduce it from another path."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    bundle=HERE/'bundle';bundle.mkdir(exist_ok=False)
    files=set((ROOT/'EDM-IR/edmir').rglob('*.py'))
    for p in (ROOT/'EDM-IR/runtime').rglob('*'):
        if p.is_file() and 'build' not in p.relative_to(ROOT).parts and (p.suffix in ('.cpp','.cu','.cuh','.hpp','.cmake') or p.name=='CMakeLists.txt'):
            files.add(p)
    for relative in (
        'EDM-IR/tools/explain_execution.py','EDM-IR/tools/prepare_execution_review.py',
        'EDM-IR/tools/predict_decision_schedule.py','EDM-IR/tools/predict_traffic_schedule.py',
        'EDM-IR/docs/EXECUTION_EVIDENCE.md','EDM-IR/pyproject.toml',
        'EDM-IR/tests/execution_evidence_test.py','EDM-IR/tests/resource_geometry_test.py',
        'EDM-IR/tests/schedule_space_test.py','EDM-IR/tests/backend_test.py',
        'experiments/2026-10-21-decision-loss-iterations/verified/models/traffic_gap.json',
        'experiments/2026-10-21-host-boundary-repair/verified/selector.json',
    ):files.add(ROOT/relative)
    # Existing geometry tests import the static candidate constructor only.
    for p in (ROOT/'experiments/2026-10-20-offline-predictor/data').glob('*'):
        if p.is_file():files.add(p)
    summary=json.loads((HERE/'review/summary.json').read_text())
    files.update(ROOT/p for p in summary['source_evidence'])
    for p in files:
        dest=bundle/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    shutil.copytree(HERE/'review',bundle/'review')
    shutil.copytree(HERE/'final-checks',bundle/'checks')
    shutil.copyfile(HERE/'verification.json',bundle/'verification.json')
    shutil.copyfile(HERE/'cli-verification.json',bundle/'cli-verification.json')
    (bundle/'README.md').write_text('''# 执行账本与证据审阅包

先读 [机制与接口说明](EDM-IR/docs/EXECUTION_EVIDENCE.md)，再看
`review/summary.json` 和 `review/mechanism_diagnosis.json`。`review/` 的五个形状文件
包含逐候选账本、历史观察和未测候选。源码位于 `EDM-IR/`。

本包没有训练或运行 GPU，也不宣称 M512/大 N 的微秒级成因已经得到验证。
证据不足返回 choice=null；这不是新的预测准确率成绩。

在解压目录执行（Python 标准库即可）：

```sh
python3 verify_bundle.py
python3 EDM-IR/tools/explain_execution.py --m 512 --k 4096 --n 512 --environment planned-a100-p4 --out /tmp/m512-execution-new.json
python3 EDM-IR/tools/prepare_execution_review.py --out /tmp/execution-review-new
python3 -m unittest discover -s EDM-IR/tests -p execution_evidence_test.py -v
```

输出路径必须不存在。完整 40 项验收还包含既有 backend/schedule/resource 检查，
其中 backend 检查需要本机 C++ 工具链；它们不执行 CUDA kernel。
旧模型及两个推断命令保留作对照，使用时另需 NumPy；不需要重新训练。
其 model_preference 仅供参考，不能代替 decision.choice。

manifest.json 固定包内文件；verification.json 记录原仓库验收及 353 个旧文件未变。
本包是本次工作的独立审阅快照，未包含那 353 个文件的全部体积、旧 GPU 输入或完整历史仓库。
''')
    verifier='''import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
expected=manifest['files']
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json' and '__pycache__' not in p.parts}
if actual!=set(expected):raise SystemExit('file set differs from manifest')
for rel,digest in expected.items():
 p=Path(rel)
 if p.is_absolute() or '..' in p.parts:raise SystemExit('unsafe manifest path')
 if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest:raise SystemExit('hash mismatch: '+rel)
print(json.dumps({'verified_files':len(expected),'status':'snapshot_identity_matches','gpu_validation':False}))
'''
    (bundle/'verify_bundle.py').write_text(verifier)
    manifest={'schema':'edmir.execution-evidence-bundle.v1','new_training_runs':0,'new_gpu_runs':0,
              'files':{str(p.relative_to(bundle)):sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=HERE/'execution-evidence-review.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,Path('execution-evidence-review')/p.relative_to(bundle))
    with tempfile.TemporaryDirectory(prefix='execution-delivery-') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(archive) as z:z.extractall(tmp)
        extracted=tmp/'execution-evidence-review'
        checked=subprocess.run([sys.executable,'-S',str(extracted/'verify_bundle.py')],capture_output=True,text=True,timeout=30)
        if checked.returncode:raise RuntimeError(checked.stderr or checked.stdout)
        reproduced=subprocess.run([sys.executable,'-S',str(extracted/'EDM-IR/tools/prepare_execution_review.py'),
                                  '--out',str(tmp/'reproduced')],cwd=extracted,capture_output=True,text=True,timeout=60)
        if reproduced.returncode:raise RuntimeError(reproduced.stderr)
        for p in (bundle/'review').glob('*.json'):
            a=json.loads(p.read_text()); b=json.loads((tmp/'reproduced'/p.name).read_text())
            if p.name=='summary.json':a.pop('elapsed_seconds');b.pop('elapsed_seconds')
            if a!=b:raise AssertionError('detached reproduction differs: '+p.name)
    delivery={'archive':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size,
              'manifest_sha256':sha(bundle/'manifest.json'),'files':len(manifest['files']),
              'detached_hash_verification':json.loads(checked.stdout),
              'detached_review_reproduced':True,'python_site_packages_required':False,
              'independent_performance_validation':False}
    (HERE/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    (HERE/(archive.name+'.sha256')).write_text(delivery['sha256']+'  '+archive.name+'\n')
    print(json.dumps(delivery,indent=2))


if __name__=='__main__':main()
