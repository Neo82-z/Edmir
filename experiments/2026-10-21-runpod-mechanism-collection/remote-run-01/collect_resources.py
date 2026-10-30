"""Capture compiled resource requirements without launching a GPU kernel."""
from pathlib import Path
import hashlib, json, sys, subprocess

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root/'bundle-r01/source/EDM-IR'), str(root/'bundle-r01/source/EDM-IR/tools')]
from run_selector_gpu import command, write_json

for phase in ('anchors', 'mechanisms', 'geometry', 'short-host-check'):
    if json.loads((root/'work'/f'{phase}.status.json').read_text())['state'] != 'complete':
        raise RuntimeError('Primary sampling must finish first')

out = root/'work/compiled-resources-02'
out.mkdir(exist_ok=False)
args = ['nvcc', '-O3', '-std=c++20', '--expt-relaxed-constexpr', '-arch=sm_80',
        '-I', root/'bundle-r01/source/EDM-IR/runtime/src',
        '-I', root/'dependencies/cutlass/include', root/'kernel_type_sizes.cu',
        '-o', out/'kernel_type_sizes']
with (out/'compile.stdout.txt').open('w') as stdout, (out/'compile.stderr.txt').open('w') as stderr:
    command(args, timeout=90, stdout=stdout, stderr=stderr)
sizes = command([out/'kernel_type_sizes'], timeout=15, capture_output=True).stdout
write_json(out/'kernel_type_sizes.json', json.loads(sizes))
binary = root/'work/build/mechanism_w004_splitk2'
with (out/'cuobjdump.txt').open('w') as stdout, (out/'cuobjdump.stderr.txt').open('w') as stderr:
    command(['cuobjdump', '--dump-resource-usage', binary], timeout=30, stdout=stdout, stderr=stderr)
with (out/'cuobjdump.txt').open() as stdin, (out/'resources-demangled.txt').open('w') as stdout:
    subprocess.run(['c++filt'], timeout=30, check=True, stdin=stdin, stdout=stdout)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
write_json(out/'receipt.json', {'command': [str(x) for x in args],
    'benchmark_binary': str(binary), 'benchmark_sha256': sha(binary),
    'introspection_source_sha256': sha(root/'kernel_type_sizes.cu'),
    'introspection_binary_sha256': sha(out/'kernel_type_sizes'),
    'scope': 'Static compiled resources, not achieved occupancy or memory traffic',
    'primary_dataset': False, 'compiler_admission': False})
print(sizes, flush=True)
