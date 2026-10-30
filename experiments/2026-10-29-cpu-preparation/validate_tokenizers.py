"""Verify local tokenizer assets with the versions selected for the GPU runtime."""
import hashlib
import json
import logging
from pathlib import Path
import time
import transformers
from transformers import AutoTokenizer

ROOT = Path('/workspace/edmir-assets')
SELECTED = {'zai-org/GLM-4.7-Flash', 'Qwen/Qwen3.6-35B-A3B',
            'deepseek-ai/DeepSeek-V2-Lite-Chat'}
rows = json.loads((ROOT / 'metadata-audit.json').read_text())['models']
results = []

class WarningCollector(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

collector = WarningCollector()
logging.getLogger('transformers').addHandler(collector)
for row in rows:
    if row['id'] not in SELECTED:
        continue
    directory = ROOT / 'hf-hub' / ('models--' + row['id'].replace('/', '--')) / 'snapshots' / row['revision']
    result = {'id': row['id'], 'revision': row['revision'], 'transformers': transformers.__version__,
              'remote_code_enabled': False, 'gpu_used': False,
              'scope': 'asset loading and template smoke only; not tokenizer equivalence or model execution'}
    collector.messages.clear()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        ids = tokenizer.encode('这是一次离线分词检查。', add_special_tokens=False)
        assert len(ids) > 0
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': '你好'}],
                                             tokenize=True, add_generation_prompt=True)
        assert len(chat) > 0
        result.update(status='passed', tokenizer_class=type(tokenizer).__name__,
                      text_tokens=len(ids), chat_tokens=len(chat), checked_unix=time.time(),
                      auxiliary_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in directory.iterdir() if p.is_file()
                                        and p.suffix in {'.json', '.jinja', '.jinja2', '.model', '.txt'}})
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, message=str(exc))
    result['warnings'] = list(collector.messages)
    results.append(result)
    print(row['id'], result['status'], flush=True)
    (ROOT / 'manifests' / 'tokenizer-checks.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
if any(row['status'] != 'passed' for row in results):
    raise SystemExit(1)
