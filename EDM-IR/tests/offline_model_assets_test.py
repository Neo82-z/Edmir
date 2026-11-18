"""Detect corrupted/index-inconsistent assets before paid GPU time."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_offline_model as audit


class AssetAuditTests(unittest.TestCase):
    def shard(self,root,*,dtype='BF16',offsets=(0,6),payload=b'\0'*6):
        header=json.dumps({'model.weight':{'dtype':dtype,'shape':[3],'data_offsets':offsets}}).encode()
        header+=b' '*(-len(header)%8)
        path=root/'model-00001.safetensors';path.write_bytes(struct.pack('<Q',len(header))+header+payload)
        return path

    def test_headers_without_full_tensor_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.shard(Path(tmp));tensors,counts,n=audit.inspect_shard(path)
            self.assertEqual(dict(counts),{'BF16':3});self.assertEqual(n,6)
            self.assertEqual(tensors['model.weight']['shape'],[3])

    def test_truncation_overlap_and_wrong_precision_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for kwargs in ({'payload':b'\0'*4},{'offsets':(2,8)},{'dtype':'F16'}):
                with self.assertRaises(ValueError):audit.inspect_shard(self.shard(root,**kwargs))

    def test_matching_bytes_with_wrong_index_still_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.shard(root)
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'wrong.weight':path.name},'metadata':{'total_size':6}}))
            manifest={'auxiliary_hashes':{index.name:audit.digest(index)},'weight_files':[
                {'rfilename':path.name,'size':path.stat().st_size,'lfs':{'sha256':audit.digest(path)}}]}
            with self.assertRaisesRegex(ValueError,'tensor headers'):audit.audit(manifest,root)

    def test_pinned_element_count_is_reported_as_warning_not_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.shard(root)
            index=root/'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map':{'model.weight':path.name},'metadata':{'total_size':3}}))
            original=index.read_bytes()
            manifest={'id':'fixture','revision':'fixed','index_reported_total_size':3,
                'auxiliary_hashes':{index.name:audit.digest(index)},'weight_files':[
                {'rfilename':path.name,'size':path.stat().st_size,'lfs':{'sha256':audit.digest(path)}}]}
            result=audit.audit(manifest,root)
            self.assertEqual(result['status'],'verified_with_metadata_warning')
            self.assertFalse(result['index_byte_size_consistent'])
            self.assertEqual(result['tensor_payload_bytes'],6)
            self.assertEqual(index.read_bytes(),original)
            manifest.pop('index_reported_total_size')
            with self.assertRaisesRegex(ValueError,'total_size'):audit.audit(manifest,root)
            index.write_text(json.dumps({'weight_map':{'model.weight':path.name},'metadata':{'total_size':4}}))
            manifest['auxiliary_hashes'][index.name]=audit.digest(index)
            manifest['index_reported_total_size']=4
            with self.assertRaisesRegex(ValueError,'total_size'):audit.audit(manifest,root)


if __name__=='__main__':unittest.main()
