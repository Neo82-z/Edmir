import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from remote import REMOTE_READER,ingest


class RemoteLogTest(unittest.TestCase):
    def read(self,path,state):
        proc=subprocess.run([sys.executable,'-c',REMOTE_READER],input=json.dumps({'glob':str(path),'files':state['files']}),
                            text=True,capture_output=True,check=True)
        return json.loads(proc.stdout)

    def test_append_resume_partial_and_rotation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'run.log';spool=root/'spool';state={'files':{}}
            source.write_bytes(b'first\npart')
            self.assertEqual(ingest('test',{'run_prefix':'test'},state,self.read(source,state),spool),1)
            self.assertEqual(state['files'][str(source)]['offset'],6)
            self.assertEqual(ingest('test',{'run_prefix':'test'},state,self.read(source,state),spool),0)
            with source.open('ab') as f:f.write(b'ial\n')
            self.assertEqual(ingest('test',{'run_prefix':'test'},state,self.read(source,state),spool),1)
            source.write_bytes(b'new\n')
            self.assertEqual(ingest('test',{'run_prefix':'test'},state,self.read(source,state),spool),1)
            self.assertEqual(state['files'][str(source)]['generation'],1)
            logs=[json.loads(line) for p in spool.glob('*.ndjson') for line in p.read_text().splitlines()]
            self.assertEqual([r['message'] for r in logs if r['event']['action']=='remote_log'],['first','partial','new'])

    def test_redaction_across_fetches(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'run.log';spool=root/'spool';state={'files':{}}
            source.write_text('password=not-for-index\n-----BEGIN PRIVATE KEY-----\n')
            ingest('test',{'run_prefix':'test'},state,self.read(source,state),spool)
            with source.open('a') as f:f.write('sensitive-block\n-----END PRIVATE KEY-----\nnormal\n')
            ingest('test',{'run_prefix':'test'},state,self.read(source,state),spool)
            output=''.join(p.read_text() for p in spool.glob('*.ndjson'))
            self.assertNotIn('not-for-index',output);self.assertNotIn('sensitive-block',output)
            self.assertIn('normal',output)

    def test_truncated_response_does_not_advance(self):
        state={'files':{}}
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                ingest('test',{'run_prefix':'test'},state,[dict(path='/tmp/run.log',start=0,end=3,reset=False,
                       identity='1:2',generation=0,size=3,content=base64.b64encode(b'abc').decode())],Path(t))
            self.assertEqual(state['files'],{})


if __name__=='__main__':unittest.main()
