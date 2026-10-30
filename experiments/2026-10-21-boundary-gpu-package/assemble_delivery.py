#!/usr/bin/env python3
"""Audit local evidence, archive the frozen bundle, and verify a relocated copy."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'EDM-IR'), str(REPO / 'EDM-IR/tools')]
from independent_selector_validation import digest
from edmir.compiler.measurement_dataset import sha256
from prepare_boundary_gpu import verify_bundle


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def check_manifest(base, entries):
    for relative, expected in entries.items():
        if sha256(base / relative) != expected:
            raise ValueError(f'prior evidence changed: {base / relative}')
    return len(entries)


def preserve_previous():
    records = []
    for name in ('2026-10-21-additive-selector-repair',
                 '2026-10-21-host-boundary-repair',
                 '2026-10-21-structural-identifiability'):
        directory = REPO / 'experiments' / name / 'verified'
        decision = json.loads((directory / 'decision.json').read_text())
        records.append({'review': name, 'decision_sha256': sha256(directory / 'decision.json'),
                        'sources_verified': check_manifest(REPO, decision['source_sha256']),
                        'evidence_files_verified': check_manifest(directory, decision['evidence_sha256'])})
    old = REPO / 'experiments/2026-10-20-local-replay-runtime-checks/gpu_pilot'
    envelope = json.loads((old / 'package.json').read_text())
    p = envelope['package']
    expected = '20e4df1a58ab838541d4212f4d99035b6312cf7ce93ae348327c138e2df4d1c5'
    assert digest(p) == envelope['package_sha256'] == expected
    source_count = check_manifest(REPO, p['sources'])
    file_count = check_manifest(old, p['files'])
    inputs = json.loads((old / 'inputs.json').read_text())
    assert inputs['package_sha256'] == expected
    for name, info in inputs['workloads'].items():
        assert sha256(old / 'data' / name / 'inputs.bin') == info['input_sha256']
        assert sha256(old / 'data' / name / 'reference.bin') == info['reference_sha256']
    return {'reviews': records, 'old_gpu_package': {
        'package_sha256': expected, 'sources_verified': source_count,
        'generated_files_verified': file_count, 'input_sets_verified': len(inputs['workloads'])}}


def main():
    bundle = HERE / 'bundle'
    p = verify_bundle(bundle)
    old = preserve_previous()
    diff = subprocess.run(['git', '-C', str(REPO), 'diff', '--check'],
                          capture_output=True, text=True, check=True, timeout=30)
    python_counts = {}
    for name, expected in [('tests.log', 6), ('base_tests.log', 11)]:
        log = (HERE / name).read_text()
        assert re.search(r'Ran (\d+) tests', log).group(1) == str(expected)
        assert log.rstrip().endswith('OK')
        python_counts[name] = expected
    assert '100% tests passed, 0 tests failed out of 72' in (HERE / 'cpp_tests.log').read_text()
    checks = json.loads((HERE / 'input_checks.json').read_text())
    assert checks['package_sha256'] == digest(p) and checks['checked_reference_elements'] == 1008
    assert len(checks['records']) == 7
    evidence_files = ['tests.log', 'base_tests.log', 'cpp_tests.log', 'cpp_build.log',
                      'cmake_configure.log', 'input_checks.json', 'prepare.log']
    test_files = [REPO / 'EDM-IR/tests' / name for name in
                  ('boundary_gpu_package_test.py', 'selector_gpu_package_test.py', 'selector_runner_safety_test.py')]
    new_tools = ['prepare_boundary_gpu.py', 'run_boundary_gpu.py',
                 'freeze_boundary_predictions.py', 'validate_boundary_inputs.py']
    source_hashes = {f'EDM-IR/tools/{name}': sha256(REPO / 'EDM-IR/tools' / name) for name in new_tools}
    source_hashes.update({str(path.relative_to(REPO)): sha256(path) for path in test_files})
    receipt = {
        'schema': 'edmir.boundary-local-verification.v1', 'date': '2026-10-21',
        'status': 'local_package_complete_gpu_validation_pending',
        'package_sha256': digest(p), 'programs': len(p['programs']),
        'process_runs_planned': len(p['jobs']), 'frozen_source_files': len(p['sources']),
        'input_and_reference_bytes': sum(f.stat().st_size for f in (bundle / 'data').rglob('*.bin')),
        'python_tests_passed': python_counts, 'cpp_cpu_tests_passed': 72,
        'independent_integer_reference_elements_checked': 1008,
        'sampled_fp16_max_abs_error': max(v['max_abs'] for r in checks['records'] for v in r['sampled_serial_fp16_errors'].values()),
        'gpu_operations': 0, 'cuda_compilation_verified': False, 'gpu_correctness_verified': False,
        'new_gpu_timing_labels': 0, 'fit_operations': 0, 'compiler_admission': False,
        'limitations': ['CPU tests use CUDA API test doubles', 'reference arithmetic is sampled',
                        'new measurements are development diagnosis, not independent validation',
                        'M512 and large-N model instability is still unresolved'],
        'local_environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'machine': platform.machine()},
        'prior_artifacts': old, 'source_sha256': source_hashes,
        'evidence_sha256': {name: sha256(HERE / name) for name in evidence_files},
        'readme_sha256': sha256(HERE / 'README.md'),
        'assembly_script_sha256': sha256(Path(__file__)), 'git_diff_check': diff.returncode == 0,
    }
    write(HERE / 'verification.json', receipt)
    archive = HERE / 'boundary-diagnosis-a100-p4.tar.gz'
    if archive.exists():
        raise FileExistsError('Use a new delivery artifact; do not overwrite an existing archive')
    prefix = 'boundary-diagnosis-a100-p4'
    def only_sources(info):
        if '__pycache__' in Path(info.name).parts or info.name.endswith(('.pyc', '.DS_Store')):
            return None
        return info
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        tar.add(bundle, arcname=f'{prefix}/bundle', filter=only_sources)
        for name in ('README.md', 'verification.json'):
            tar.add(HERE / name, arcname=f'{prefix}/{name}')
        for name in evidence_files:
            tar.add(HERE / name, arcname=f'{prefix}/evidence/{name}')
        for path in test_files:
            tar.add(path, arcname=f'{prefix}/evidence/source_tests/{path.name}')
    archive_hash = sha256(archive)
    (HERE / (archive.name + '.sha256')).write_text(f'{archive_hash}  {archive.name}\n')
    with tempfile.TemporaryDirectory(prefix='edmir-boundary-transfer-') as temp:
        with tarfile.open(archive) as tar:
            tar.extractall(temp, filter='data')
        relocated = Path(temp) / prefix
        # The source and data must work without the checkout or site-packages.
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable, '-B', '-S',
                                 str(relocated / 'bundle/source/EDM-IR/tools/prepare_boundary_gpu.py'),
                                 'verify', '--package', str(relocated / 'bundle')],
                                cwd=temp, env=env, capture_output=True, text=True, check=True, timeout=60)
        transferred = json.loads(result.stdout)
        assert transferred['package_sha256'] == digest(p)
        for name in ('README.md', 'verification.json'):
            assert sha256(relocated / name) == sha256(HERE / name)
        for name in evidence_files:
            assert sha256(relocated / 'evidence' / name) == sha256(HERE / name)
        for path in test_files:
            assert sha256(relocated / 'evidence/source_tests' / path.name) == sha256(path)
    delivery = {'archive': archive.name, 'archive_sha256': archive_hash,
                'archive_bytes': archive.stat().st_size, 'package_sha256': digest(p),
                'relocated_without_checkout_verified': True, 'site_packages_disabled': True,
                'bundled_evidence_verified': True, 'verification_sha256': sha256(HERE / 'verification.json'),
                'gpu_validated': False}
    write(HERE / 'delivery.json', delivery)
    print(json.dumps(delivery, indent=2))


if __name__ == '__main__':
    main()
