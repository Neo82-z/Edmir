#!/usr/bin/env python3
"""Verify the local ELK path, permissions and saved views; never contact a cloud."""
import argparse
import base64
import fcntl
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from ops import emit, now
from setup import LOCAL, ROOT, credentials, request


def as_reader(path, body=None, kibana=False):
    auth = base64.b64encode(('edmir:' + credentials()['EDMIR_PASSWORD']).encode()).decode()
    req = urllib.request.Request(
        ('http://127.0.0.1:15601' if kibana else 'http://127.0.0.1:19200') + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': 'Basic ' + auth, 'Content-Type': 'application/json', 'kbn-xsrf': 'edmir-verify'},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode()


def compose(*args):
    subprocess.run(
        ['docker-compose', '--env-file', '.local/stack.env', '-f', 'compose.yaml', *args],
        cwd=ROOT, env={**os.environ, 'DOCKER_CONTEXT': 'colima-edmir-logs'}, check=True,
        timeout=90,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-restart', action='store_true')
    args = parser.parse_args()
    checks = {}
    if args.check_restart:
        compose('stop', 'logstash')
        try:
            key = 'restart-' + uuid.uuid4().hex
            for _ in range(2):
                emit(LOCAL/'events', 'log-center-acceptance', 'ingestion_restart_check',
                     '本机 Logstash 停止期间落盘；同一事件重复投递，用于验证补入与去重。',
                     outcome='success', kind='local_verification', key=key)
        finally:
            compose('up', '-d', 'logstash')
    # Capture a complete-line snapshot while live SSH forwarding keeps appending.
    expected = {}
    for path in (LOCAL/'events').glob('*.ndjson'):
        with path.open() as stream:
            fcntl.flock(stream,fcntl.LOCK_SH)
            for row in map(json.loads,stream.read().splitlines()):expected[row['event']['id']]=row
    deadline = time.monotonic() + 90
    missing = set(expected)
    while missing and time.monotonic() < deadline:
        missing = set()
        ids = list(expected)
        for offset in range(0, len(ids), 200):
            batch = ids[offset:offset+200]
            result = request('/edmir-ops-*/_search', {'size':200, '_source':False, 'query':{'ids':{'values':batch}}})
            missing.update(set(batch)-{hit['_id'] for hit in result['hits']['hits']})
            if len(result['hits']['hits'])==len(batch):
                assert result['hits']['total']['value']==len(batch), 'Duplicated snapshot IDs across indices'
        if missing:
            time.sleep(2)
    assert not missing, f'{len(missing)} journal events not indexed'
    count = request('/edmir-ops-*/_count')['count']
    checks['all_journal_event_ids_indexed'] = len(expected)
    checks['index_documents'] = count
    assert count >= len(expected), 'Index contains fewer documents than the journal snapshot'
    checks['documents_added_after_journal_snapshot'] = count-len(expected)
    checks['duplicate_delivery_idempotent'] = True
    checks['restart_backlog_checked_this_run'] = args.check_restart

    try:
        urllib.request.urlopen('http://127.0.0.1:19200/edmir-ops-*/_count', timeout=10)
        raise AssertionError('Elasticsearch allowed unauthenticated reads')
    except urllib.error.HTTPError as e:
        assert e.code == 401
    checks['unauthenticated_access_denied'] = True
    assert json.loads(as_reader('/edmir-ops-*/_count'))['count'] >= count
    privileges = json.loads(as_reader('/_security/user/_has_privileges', {
        'index':[{'names':['edmir-ops-*'],'privileges':['read','write','delete_index']}]}
    ))['index']['edmir-ops-*']
    assert privileges['read'] and not privileges['write'] and not privileges['delete_index']
    checks['reader_index_permissions'] = privileges
    view = json.loads(as_reader('/api/data_views/data_view/edmir-operations', kibana=True))['data_view']
    assert view['timeFieldName'] == '@timestamp'
    export = as_reader('/api/saved_objects/_export', {
        'objects':[{'type':'dashboard','id':'edmir-operations-dashboard'}],
        'includeReferencesDeep':True, 'excludeExportDetails':True,
    }, kibana=True)
    objects = [json.loads(line) for line in export.splitlines() if line.strip()]
    identities = {(o['type'],o['id']) for o in objects}
    required = {('dashboard','edmir-operations-dashboard'),('visualization','edmir-intro'),
                ('index-pattern','edmir-operations')} | {('search','edmir-'+n) for n in ('all','failures','recovery','commands')}
    assert required <= identities
    checks['saved_objects_verified_as_reader'] = len(required)
    (LOCAL/'saved-objects.ndjson').write_text(export)
    health = request('/_cluster/health')
    assert health['status'] == 'green'
    kibana = request('/api/status', kibana=True)['status']['overall']['level']
    assert kibana == 'available'
    checks['elasticsearch'] = health['status']
    checks['kibana'] = kibana
    checks['visual_browser_review'] = 'not performed; API verification only'
    report = {'verified_at':now(), 'checks':checks}
    (LOCAL/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
