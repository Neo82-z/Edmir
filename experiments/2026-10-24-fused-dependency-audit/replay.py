"""Kernel precedence from actual CPU-stub API calls, not a latency simulator."""
from collections import Counter

def read_capture(path):
    order=[]; sections={}; section=None
    for line in path.read_text().splitlines():
        if line.startswith('ORDER '): order.append(tuple(map(int,line.split()[1:])))
        elif line=='SETUP' or line.startswith('ITER '):
            section=line;sections[section]=[]
        else: sections[section].append(line)
    return order,sections

def replay(lines,order):
    streams={};events={};host=set();ancestors={};launches=[];counts=Counter()
    # Each captured iteration starts after a full synchronize of the previous
    # one. Previous event generations are therefore completed, not missing data.
    for line in lines:
        a=line.split();name=a[0];counts[name]+=1;args=a[2:]
        if name=='wait_event':
            stream,event,_=args
            if event not in events:raise ValueError('wait before current-iteration record')
            streams.setdefault(stream,set()).update(events[event])
        elif name=='record_event':
            event,stream=args;events[event]=set(streams.get(stream,set()))|host
        elif name=='fused_cutlass':
            op,src,dst=order[len(launches)];stream=args[-1]
            assert int(args[3])==src
            ancestors[op]=set(streams.get(stream,set()))|host
            assert op not in ancestors[op]
            streams[stream]=ancestors[op]|{op};launches.append(op)
        elif name=='sync_stream':host.update(streams.get(args[0],set()))
        elif name!='set_device':raise ValueError('unsupported API in fused execution: '+name)
    assert len(launches)==len(order) and len(host)==len(order)
    return ancestors,dict(counts)

def paths(order,ancestors):
    chains={}
    for op,_,_ in order:
        chains[op]=max((chains[p] for p in ancestors[op]),key=len,default=[])+[op]
    return max(chains.values(),key=len)

def closure(order,edges):
    anc={}
    for op,_,_ in order:
        before=[a for a,b in edges if b==op]
        assert all(a in anc for a in before)
        anc[op]=set(before).union(*(anc[a] for a in before))
    return anc
