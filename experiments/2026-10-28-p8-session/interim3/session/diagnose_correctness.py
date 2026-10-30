#!/usr/bin/env python3
"""Separate post-collection diagnostic binary; no tolerance or kernel changes."""
from pathlib import Path
import os,json,time,hashlib,shlex,subprocess
root=Path('/root/2026-10-25-p8-gpu-collection');build=root/'work/build';out=root/'work/correctness-diagnostic'
DEADLINE=1793165400
while not (root/'work/subset/final.json').exists() and time.time()<DEADLINE-100:time.sleep(5)
if not (root/'work/subset/final.json').exists():raise SystemExit('Main collection has priority; diagnostic skipped.')
if time.time()>DEADLINE-100:raise SystemExit('Insufficient reserved time; diagnostic skipped.')
out.mkdir(exist_ok=False)
source=root/'bundle/source/EDM-IR/runtime/tools/benchmark_selector.cpp'
s=source.read_text();old='if (!before.passed()) throw std::runtime_error("initial numerical correctness failed");'
new='''std::cerr << std::setprecision(17) << "{\\"phase\\":\\"initial\\",\\"passed\\":" << (before.passed() ? "true" : "false") << ",\\"max_abs_error\\":" << before.absolute << ",\\"max_rel_error\\":" << before.relative << ",\\"checked_elements\\":" << before.checked << ",\\"nonfinite\\":" << before.nonfinite << ",\\"atol\\":0.02,\\"rtol\\":0.02}\\n";
  if (!before.passed()) throw std::runtime_error("initial numerical correctness failed");'''
assert s.count(old)==1;s=s.replace(old,new)
src=out/'benchmark_selector_diagnostic.cpp';src.write_text(s)
target='p8_w005_splitk16_immutable';flags={}
for line in (build/f'CMakeFiles/{target}.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);flags[k]=shlex.split(v)
obj=out/'diagnostic.o';exe=out/'diagnostic'
compile=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(src),'-o',str(obj)]
link=shlex.split((build/f'CMakeFiles/{target}.dir/link.txt').read_text());link=[str(obj) if x==f'CMakeFiles/{target}.dir/tools/benchmark_selector.cpp.o' else str(exe) if x==target else x for x in link]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'purpose':'Expose metrics missing from preserved numerical failure; no threshold/kernel/shape changes. Diagnostic is a separate binary and excluded from timing comparisons.','original_source_sha256':sha(source),'diagnostic_source_sha256':sha(src),'compile':compile,'link':link,'created_epoch':time.time(),'tests':[]}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (out/'build.stdout.txt').open('w') as so,(out/'build.stderr.txt').open('w') as se:
 subprocess.run(compile,cwd=build,stdout=so,stderr=se,check=True,timeout=30)
 subprocess.run(link,cwd=build,stdout=so,stderr=se,check=True,timeout=30)
receipt['executable_sha256']=sha(exe)
for i in range(3):
 if time.time()>DEADLINE-65:break
 args=[str(exe),'splitk16',str(root/'bundle/data/w005/inputs.bin'),str(root/'bundle/data/w005/reference.bin'),'0','1','prepared']
 start=time.time()
 with (out/f'stdout_{i}.txt').open('w') as so,(out/f'stderr_{i}.txt').open('w') as se:
  try:r=subprocess.run(args,stdout=so,stderr=se,timeout=60);result={'returncode':r.returncode}
  except subprocess.TimeoutExpired:result={'timeout':True}
 receipt['tests'].append({'repetition':i,'args':args,'elapsed_seconds':time.time()-start,**result})
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
receipt['complete_epoch']=time.time();(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
