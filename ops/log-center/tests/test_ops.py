import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops import emit,redact,redact_argv,run_command,import_w1,sha

class OperationsTest(unittest.TestCase):
    def rows(self,root):return [json.loads(line) for f in root.glob('*.ndjson') for line in f.read_text().splitlines()]
    def test_redaction(self):
        self.assertNotIn('abc123',redact('Authorization: Bearer abc123'))
        self.assertNotIn('secret123',redact_argv(['command','--api-key','secret123']))
        self.assertEqual(redact({'token':'hidden','normal':3}),{'token':'[REDACTED]','normal':3})
        self.assertNotIn('hunter2',redact('{"password": "hunter2"}'))
    def test_timestamp_and_stable_id(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            a=emit(p,'test','test','a',key='same',ts='2026-11-11T16:00:00+08:00')
            b=emit(p,'test','test','a',key='same')
            self.assertEqual(a,b);self.assertTrue(self.rows(p)[0]['@timestamp'].startswith('2026-11-11T08:00'))
            with self.assertRaises(ValueError):emit(p,'t','test','bad',ts='2026-11-11T16:00:00')
    def test_real_exit_and_timeout(self):
        with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            p=Path(t)
            self.assertEqual(run_command(p,'test',[sys.executable,'-c','print("hello");raise SystemExit(7)'],5),7)
            self.assertEqual(run_command(p,'test',[sys.executable,'-c','import time;time.sleep(30)'],0.15),124)
            end=[r for r in self.rows(p) if r['event']['action']=='command_finished']
            self.assertEqual([r['process']['exit_code'] for r in end],[7,124])
            self.assertTrue(all(r['event']['outcome']=='failure' for r in end))
    def test_dedup_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);s=p/'session';r=s/'recovered';(r/'results').mkdir(parents=True);(r/'session').mkdir()
            (s/'session-plan.json').write_text(json.dumps({'active_at_utc':'2026-11-11T07:29:14Z'}))
            artifact=r/'artifact.txt';artifact.write_text('original')
            (r/'recovery-manifest.json').write_text(json.dumps({'files':{'artifact.txt':{'sha256':sha(artifact),'bytes':8}}}))
            spool=p/'spool';self.assertGreater(import_w1(spool,s,'test'),0)
            self.assertEqual(import_w1(spool,s,'test'),0)
            artifact.write_text('tampered')
            self.assertEqual(import_w1(spool,s,'test'),2)
            failures=[r for r in self.rows(spool) if r['event']['outcome']=='failure']
            self.assertEqual(len(failures),2)
    def test_spawn_failure_has_terminal_record(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.assertEqual(run_command(p,'test',['/no/such/command'],1),1)
            self.assertEqual(self.rows(p)[-1]['event']['action'],'command_finished')
if __name__=='__main__':unittest.main()
