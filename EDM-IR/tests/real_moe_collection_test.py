"""CPU-only adversarial tests; generated records are always synthetic_test."""
import asyncio
from copy import deepcopy
from contextlib import closing
import gzip
import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from edmir.benchmark.real_moe import (StepSampler, input_rows, slice_routes, summarize_routes,
    collect_block, timeline_overlap, validate_corpus, tokenize_corpus, validate_token_packet,
    schedule, JsonlWriter, read_journal, digest, summarize_blocks, model_vocab_size)
from edmir.benchmark.vllm0201_capture import padded_assignment, route_block, verify_sources
from edmir.benchmark.real_moe_config import engine_arguments, validate_topology
from build_real_moe_corpus import build
from run_real_moe import check_guard
from analyze_real_moe import nsys_inventory, nsys_intervals


def fixture(category="mixed",geometry=False):
    # 2 real source ranks and 6 empty/dummy source ranks; all 46 MoE layers.
    tokens=[input_rows([dict(request_id=f"diag-S1-{r:03d}",computed_tokens=1 if category!='pure_decode' else 2,
        scheduled_tokens=2 if category=='mixed' else 1,prompt_tokens=2)]) if r<2 else [] for r in range(8)]
    ids=[[[0,8,16,24] for _ in t] for t in tokens];weights=[[[.25]*4 for _ in t] for t in tokens]
    events=[]
    for rank in range(8):
        base=dict(origin="synthetic_test",run_id="fixture",block_id="diag-S1",step=1,rank=rank)
        events.append(dict(**base,kind="step",tokens=tokens[rank],category=category,selected=True))
        for layer in range(1,47):
            result=slice_routes(ids[rank],weights[rank],rank=rank,sizes=[len(t) for t in tokens],tokens=tokens[rank],layout="local")
            events.append(dict(**base,kind="route",layer=layer,**result))
            if geometry:
                emap=[i-rank*8 if rank*8<=i<(rank+1)*8 else -1 for i in range(64)]
                n=sum(map(len,tokens)) if rank<4 else 0
                slots={0:dict(real_rows=n,block_slots=32,dummy_or_padding_slots=32-n)} if n else {}
                for p,shape in (("gate_up",[8,3072,2048]),("down",[8,2048,1536])):
                    events.append(dict(**base,kind="kernel",layer=layer,projection=p,weight_shape=shape,
                        input_dtype="torch.bfloat16",weight_dtype="torch.bfloat16",physical_layout_verified=True,
                        expert_map=emap,local_expert_assignment=slots))
    return events


class FakeTokenizer:
    """Character IDs test the slicing protocol only, never real GLM lengths."""
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def apply_chat_template(self,messages,tokenize=True,**kwargs):
        text="<user>"+messages[0]['content']+"</user><assistant>"
        if not tokenize:return text
        ids=self.encode(text)
        # Transformers 5 defaults to a mapping; the collector requests IDs.
        return {'input_ids':ids,'attention_mask':[1]*len(ids)} if kwargs.get('return_dict',True) else ids


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.corpus=build()
    def test_content_freeze_and_schedule(self):
        self.assertEqual(validate_corpus(self.corpus)['source_requests'],128)
        self.assertEqual(build()['request_content_sha256'],self.corpus['request_content_sha256'])
        rows=schedule();self.assertEqual(sum(r['requests'] for r in rows),960)
        self.assertEqual(sum(r['requests']*r['input_tokens'] for r in rows),2211840)
        self.assertEqual(sum(r['requests']*r['output_tokens'] for r in rows),122880)
        bad=deepcopy(self.corpus);bad['requests'][0]['content']+=' changed'
        with self.assertRaises(ValueError):validate_corpus(bad)
    def test_exact_length_and_template_protocol_with_fake_tokenizer(self):
        packet=tokenize_corpus(self.corpus,FakeTokenizer())
        self.assertTrue(validate_token_packet(packet,self.corpus,0x110000))
        self.assertEqual(len(packet['requests']),256)
        packet['requests'][0]['prompt_token_ids'][-1]+=1
        with self.assertRaises(ValueError):validate_token_packet(packet,self.corpus,0x110000)
    def test_vocab_size_of_every_locked_model_config(self):
        # Qwen3.5 is multimodal and keeps vocab_size under text_config; the others have it at the top.
        models=ROOT.parent/'experiments/2026-11-20-final-round/models'
        with tempfile.TemporaryDirectory() as d:
            for name,expected in (('GLM',154880),('QWEN35',248320),('DSV2L',102400)):
                (Path(d)/'config.json').write_text((models/f'{name}.config.json').read_text())
                self.assertEqual(model_vocab_size(d),expected)
            (Path(d)/'config.json').write_text(json.dumps(dict(text_config=dict(hidden_size=8))))
            with self.assertRaises(ValueError):model_vocab_size(d)
    def test_config_preserves_baseline_and_diagnostic_separation(self):
        baseline=engine_arguments('/offline/model','baseline');routes=engine_arguments('/offline/model','routes')
        self.assertFalse(baseline['enforce_eager']);self.assertNotIn('worker_cls',baseline)
        self.assertTrue(routes['enforce_eager']);self.assertFalse(routes['async_scheduling'])
        self.assertEqual(baseline['max_model_len'],4608)
        self.assertEqual(baseline['tensor_parallel_size']*baseline['data_parallel_size'],8)


class RouterTests(unittest.TestCase):
    def test_mixed_tokens_and_source_dedup_after_allgather(self):
        t=input_rows([dict(request_id='diag-S1-000',computed_tokens=1,prompt_tokens=2,scheduled_tokens=2)])
        self.assertEqual([x['phase'] for x in t],['prefill','decode'])
        ids=[[0,1,2,3]]*12;weights=[[.25]*4]*12
        result=slice_routes(ids,weights,rank=1,sizes=[3,3,1,1,1,1,1,1],tokens=t,layout='gathered_dp')
        self.assertEqual(result['source_slice_offset'],3);self.assertEqual(result['token_count'],2)
        self.assertEqual(sum(result['phase_counts']['prefill']),4)
        with self.assertRaises(ValueError):slice_routes(ids,weights,rank=1,sizes=[3,3,1,1,1,1,1,1],tokens=t,layout='local')
        bad=deepcopy(ids);bad[3]=[0,0,1,2]
        with self.assertRaises(ValueError):slice_routes(bad,weights,rank=1,sizes=[3,3,1,1,1,1,1,1],tokens=t,layout='gathered_dp')
    def test_weights_finiteness_and_no_fake_zero_rows(self):
        t=[dict(request_id='x',position=0,phase='decode')]
        with self.assertRaises(ValueError):slice_routes([[0,1,2,3]],[[float('nan')]*4],rank=0,sizes=[1]*8,tokens=t,layout='local')
        z=slice_routes([[0,1,2,3]],[[.25]*4],rank=0,sizes=[1]*8,tokens=[],layout='local')
        self.assertEqual(z['token_count'],0);self.assertEqual(sum(z['phase_counts']['decode']),0)
    def test_sampler_bounds_global_category(self):
        sampler=StepSampler()
        rows=[dict(rank=i,block_id='diag-S1',prefill=1 if i==0 else 0,decode=1 if i==1 else 0) for i in range(8)]
        self.assertEqual([sampler.observe(s,rows)[1] for s in range(5)],[True,True,True,False,False])
        self.assertEqual(sampler.steps['mixed'],5)
        with self.assertRaises(ValueError):sampler.observe(6,rows[:-1])
        rows[1]['block_id']='other'
        with self.assertRaises(ValueError):sampler.observe(7,rows)
    def test_full_rank_layer_conservation_and_permutation(self):
        f=fixture(geometry=True);a=summarize_routes(f)
        self.assertEqual(a['sampled_routed_rows'],4*4*46)
        self.assertTrue(a['kernel_geometry_complete']);self.assertEqual(a['origin'],'synthetic_test')
        random.Random(42).shuffle(f);b=summarize_routes(f)
        self.assertEqual(a,b)
    def test_missing_rank_layer_and_duplicated_tokens_rejected(self):
        f=fixture();cases=[]
        cases.append([r for r in f if r['rank']!=7])
        cases.append([r for r in f if not (r['rank']==1 and r.get('layer')==46)])
        cases.append(f+[deepcopy(next(r for r in f if r['kind']=='route'))])
        bad=deepcopy(f)
        next(r for r in bad if r['kind']=='step' and r['rank']==1)['tokens']=deepcopy(bad[0]['tokens']);cases.append(bad)
        for rows in cases:
            with self.subTest(case=len(rows)),self.assertRaises(ValueError):summarize_routes(rows)
    def test_corrupt_counts_and_forged_geometry_rejected(self):
        f=fixture();next(r for r in f if r['kind']=='route')['phase_counts']['prefill'][0]+=1
        with self.assertRaises(ValueError):summarize_routes(f)
        f=fixture(geometry=True);next(r for r in f if r['kind']=='kernel')['weight_shape']=[8,3072,1024]
        with self.assertRaises(ValueError):summarize_routes(f)
        f=fixture(geometry=True);next(r for r in f if r['kind']=='kernel')['local_expert_assignment'][0]['real_rows']+=1
        with self.assertRaises(ValueError):summarize_routes(f)
    def test_missing_geometry_is_unknown_not_imputed(self):
        a=summarize_routes(fixture());self.assertFalse(a['kernel_geometry_complete'])
        self.assertIsNone(a['measured_time_coverage'])
    def test_actual_assignment_masks_dummy_and_padding(self):
        # Two router rows, second is dummy. Four live top-k assignments total.
        r=padded_assignment([0,1,2,3,4,5,8,8],[0,1],8,4,[True,False])
        self.assertEqual(r[0]['real_rows'],4);self.assertEqual(r[1]['real_rows'],0)
        self.assertEqual(r[1]['dummy_or_padding_slots'],4)
        with self.assertRaises(ValueError):padded_assignment([0,0,2,3],[0],4,4,[True])
    def test_route_block_and_journal_bound(self):
        self.assertEqual(route_block([dict(request_id='diag-S1-000')]),'diag-S1')
        self.assertIsNone(route_block([dict(request_id='warmup-diag-S1-0-000')]))
        with self.assertRaises(ValueError):route_block([dict(request_id='unknown')])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.gz';w=JsonlWriter(p,limit=40);w.write({'origin':'synthetic_test'})
            with self.assertRaises(ValueError):w.write({'large':'x'*100})
            w.close();self.assertEqual(len(list(read_journal(p))),1)


class AsyncCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_closed_loop_cap_chunk_timing_and_failure_retention(self):
        active=peak=0
        async def generate(r,rid):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            try:
                await asyncio.sleep(.001)
                if r['source_id']=='bad':raise RuntimeError('fixture failure')
                yield dict(token_ids=list(range(64)),prompt_token_ids=r['prompt_token_ids'],finished=False)
                await asyncio.sleep(.001)
                yield dict(token_ids=list(range(128)),prompt_token_ids=r['prompt_token_ids'],finished=True)
            finally:active-=1
        req=[dict(source_id=str(i),prompt_token_ids=[1,2]) for i in range(8)];req[2]['source_id']='bad'
        r=await collect_block(generate,req,block_id='fixture',concurrency=2)
        self.assertEqual(peak,2);self.assertEqual(r['successes'],7);self.assertEqual(len(r['requests']),8)
        self.assertIsNone(r['requests'][0]['itl_seconds']);self.assertIn('fixture failure',r['requests'][2]['error'])
    async def test_timeout_and_prompt_mismatch(self):
        async def slow(r,rid):
            await asyncio.sleep(.03)
            yield dict(token_ids=[],prompt_token_ids=r['prompt_token_ids'],finished=True)
        r=await collect_block(slow,[dict(source_id='x',prompt_token_ids=[1])],block_id='timeout',concurrency=1,timeout=.001)
        self.assertEqual(r['successes'],0)
        async def changed(r,rid):yield dict(token_ids=list(range(128)),prompt_token_ids=[2],finished=True)
        r=await collect_block(changed,[dict(source_id='x',prompt_token_ids=[1])],block_id='changed',concurrency=1)
        self.assertIn('changed prompt',r['requests'][0]['error'])
    async def test_cancellation_writes_partial_receipt(self):
        async def gen(r,rid):
            if r['source_id']=='slow':await asyncio.sleep(10)
            yield dict(token_ids=list(range(128)),prompt_token_ids=r['prompt_token_ids'],finished=True)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'partial.json'
            task=asyncio.create_task(collect_block(gen,[dict(source_id=s,prompt_token_ids=[1]) for s in ('fast','slow')],
                block_id='cancel',concurrency=2,checkpoint=path))
            await asyncio.sleep(.01);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            receipt=json.loads(path.read_text());self.assertEqual(len(receipt['completed_requests']),1)
            self.assertEqual(len(receipt['unfinished_request_ids']),1)


class AnalysisAndGuardTests(unittest.TestCase):
    def test_overlap_uses_interval_union_and_separate_clocks(self):
        rows=[dict(origin='synthetic_test',rank=0,clock_domain='a',kind=k,start_ns=s,end_ns=e)
              for k,s,e in [('communication',0,10),('communication',5,15),('compute',7,20)]]
        r=timeline_overlap(rows)['ranks'][0]
        self.assertEqual(r['communication_union_ns'],15);self.assertEqual(r['overlap_ns'],8)
        rows.append(dict(origin='synthetic_test',rank=0,clock_domain='b',kind='compute',start_ns=0,end_ns=100))
        self.assertEqual(len(timeline_overlap(rows)['ranks']),2)
    def test_gpu_guard_rejects_missing_stale_or_overbudget(self):
        g=dict(schema='edmir.real-moe-gpu-guard.v1',gpu_execution_authorized=True,
            platform_auto_delete_verified=True,instance_id='test',billing_start_epoch=1000,
            platform_delete_epoch=4600,usd_per_hour=10,approved_usd=15)
        self.assertEqual(check_guard(g,1100,'test'),4480)
        for bad,now,host in [({},1100,'test'),(g,4500,'test'),(g,1100,'other'),({**g,'usd_per_hour':16},1100,'test')]:
            with self.assertRaises(ValueError):check_guard(bad,now,host)
    def test_source_binding_and_trace_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.py';p.write_text('x=1\n')
            from edmir.benchmark.real_moe import file_sha
            lock={'files':{'source.py':file_sha(p)}};verify_sources(root,lock);p.write_text('x=2\n')
            with self.assertRaises(ValueError):verify_sources(root,lock)
            dbpath=root/'trace.sqlite'
            with closing(sqlite3.connect(dbpath)) as db, db:
                db.execute('create table StringIds(id integer,value text)')
                db.execute('insert into StringIds values(1,"unknown kernel")')
                db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(globalPid integer,deviceId integer,start integer,end integer,demangledName integer)')
                db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(123,0,0,10,1)')
            self.assertEqual(nsys_inventory(dbpath)[0]['kernels'],1)
            with self.assertRaises(ValueError):nsys_intervals(dbpath,{}, {})
            records=nsys_intervals(dbpath,{'123:0':{'rank':0,'evidence':'synthetic worker map'}},{})
            self.assertEqual(records[0]['kind'],'unknown')
    def test_partial_performance_never_passes(self):
        r=summarize_blocks([]);self.assertFalse(r['complete']);self.assertEqual(len(r['missing_blocks']),12)
    def test_pcie_cannot_masquerade_as_nvlink(self):
        text='\n'.join('GPU'+str(i)+' '+ ' '.join('X' if i==j else 'NV12' for j in range(8)) for i in range(8))
        self.assertEqual(len(validate_topology(text)),8)
        with self.assertRaises(ValueError):validate_topology(text.replace('NV12','PHB',1))
        with self.assertRaises(ValueError):validate_topology('\n'.join(text.splitlines()[:-1]))


if __name__=='__main__':unittest.main()
