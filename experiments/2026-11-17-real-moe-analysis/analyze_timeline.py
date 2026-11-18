#!/usr/bin/env python3
"""Inventory only. Profiled/failed windows cannot establish serving speedup."""
from pathlib import Path
import json,sqlite3
HERE=Path(__file__).resolve().parent
RAW=HERE.parent/'2026-11-17-real-moe-gpu/recovered'
out=[]
for f in sorted(RAW.glob('*.sqlite')):
 c=sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 def rows(sql):return [dict(r) for r in c.execute(sql)]
 d={'file':f.name,'session_start':rows('select * from TARGET_INFO_SESSION_START_TIME'),'bounds':rows('select min(start) first_kernel_ns,max(end) last_kernel_ns,count(*) kernels from CUPTI_ACTIVITY_KIND_KERNEL'),'profiler_api':rows("select r.start,r.end,r.globalTid,s.value,r.returnValue from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like '%Profiler%'"),'nvtx':rows('select text,count(*) n,min(start) first_ns,max(end) last_ns from NVTX_EVENTS group by text order by count(*) desc limit 20'),'top_kernels':rows('select s.value,count(*) n,sum(k.end-k.start)*1e-9 summed_device_seconds from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id group by k.shortName order by sum(k.end-k.start) desc limit 16'),'diagnostics':rows('select * from DIAGNOSTIC_EVENT'),'analysis_details':rows('select * from ANALYSIS_DETAILS')}
 out.append(d); print(f.name,d['bounds'],flush=True)
(HERE/'timeline-analysis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
