"""CPU counterfactual only: precision at a recorded numerical error witness."""
from pathlib import Path
import json,struct
ROOT=Path(__file__).resolve().parent
w=json.loads((ROOT.parent/'2026-10-28-p8-session/witness-reproduction.json').read_text())['witnesses'][0]
parts=w['exact_partition_sums'];exact=sum(parts)
def bf16(x):
 bits=struct.unpack('<I',struct.pack('<f',x))[0]
 bits=((bits+0x7fff+((bits>>16)&1))>>16)<<16
 return struct.unpack('<f',struct.pack('<I',bits))[0]
def fp16(x):return struct.unpack('<e',struct.pack('<e',x))[0]
rows=[]
for dtype,rounder in [('fp16',fp16),('bf16',bf16)]:
 for partial in (dtype,'fp32'):
  acc=0.
  for v in parts:acc=rounder(acc+v) if partial==dtype else acc+v
  actual=rounder(acc);rows.append({'input':dtype,'partition_accumulator':'fp32','partial':partial,'output':dtype,'simulated_result':actual,'error':abs(actual-exact)})
report={'kind':'CPU numerical counterfactual on one observed FP16 error witness','reference':exact,
 'input_note':'Original integer/32 inputs are exactly representable in FP16 and BF16; this witness isolates intermediate rounding. No BF16 GPU latency or whole-kernel correctness is established.',
 'rounding':'round-to-nearest ties-to-even; exact FP32 dyadic partition sums for this input only','results':rows}
(ROOT/'precision-witness.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 import torch
 # Independently check the software BF16 rounding against PyTorch conversion.
 for value in parts+[exact,1.00390625,1.01171875,-1.00390625]:
  assert bf16(value)==torch.tensor(value,dtype=torch.float32).to(torch.bfloat16).float().item()
 print(json.dumps(report,indent=2))
