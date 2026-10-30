#!/usr/bin/env python3
"""Plot the fixed-K/N M sweep; process medians are observations, not CIs."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
source=HERE/'review-primary-local.json'
review=json.loads(source.read_text())
rows=[r for s in review['stages'] if s['stage'] in ('anchors','crossings') for r in s['configurations']]
families=['splitk2@prepared','splitk4_groupedn2@prepared','splitk8_groupedn2@prepared']
fig,axes=plt.subplots(1,3,figsize=(12.6,4.4))
data=[]
for ax,cid,title in zip(axes,families,['q2, g1','q4, g2','q8, g2']):
    rs=sorted([r for r in rows if r['candidate_id']==cid and r['shape'][2:]==[8192,512] and r['shape'][1] in (192,256,320)],key=lambda r:r['shape'][1])
    assert len(rs)==3 and not any(r['quality_flags'] for r in rs)
    xs=[r['shape'][1] for r in rs]
    for x,r in zip(xs,rs):
        ax.scatter([x]*3,r['process_medians_us'],s=23,color='#777777',alpha=.65,zorder=3)
        data.append({'shape':r['shape'],'candidate_id':cid,'process_medians_us':r['process_medians_us'],
            'observed_us':r['observed_us'],'forecasts':r['forecast_errors']})
    ax.plot(xs,[r['observed_us'] for r in rs],'-o',color='#222222',label='Measured median',lw=2,ms=4)
    ax.plot(xs,[r['forecast_errors']['regime_max']['predicted_us'] for r in rs],'--s',color='#2166ac',label='regime_max',lw=2,ms=4)
    ax.plot(xs,[r['forecast_errors']['residual_depth']['predicted_us'] for r in rs],':^',color='#d95f02',label='residual_depth',lw=2,ms=4)
    ax.set_title(title,fontsize=12);ax.set_xticks(xs);ax.set_xlabel('M (rows per peer)')
    ax.set_ylabel('Latency (microseconds)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8,loc='upper left',frameon=False)
fig.suptitle('Frozen forecasts versus fresh A100 measurements',fontsize=15,y=.98)
fig.text(.5,.025,'4 x A100 SXM4 80GB, NV12 | K = 8192, N = 512 | Dots: three process medians, not confidence intervals\nHost execute + synchronize timing; no refit; targeted diagnostic shapes.',ha='center',fontsize=8,color='#444444')
fig.tight_layout(rect=(0,.10,1,.94))
fig.savefig(HERE/'m-sweep.png',dpi=180)
fig.savefig(HERE/'m-sweep.svg')
(HERE/'m-sweep-data.json').write_text(json.dumps({'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'matplotlib_version':matplotlib.__version__,'rows':data},indent=2)+'\n')
