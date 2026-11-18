#!/usr/bin/env python3
"""Prepare local credentials, ECS mappings and Kibana views. No cloud actions."""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parent
LOCAL=ROOT/'.local'

def credentials():
    return dict(line.split('=',1) for line in (LOCAL/'stack.env').read_text().splitlines() if '=' in line)

def init():
    LOCAL.mkdir(mode=0o700,exist_ok=True)
    (LOCAL/'events').mkdir(exist_ok=True)
    path=LOCAL/'stack.env'
    if path.exists():
        print('Existing local credentials retained');return
    values={k:secrets.token_hex(24) for k in ('ELASTIC_PASSWORD','KIBANA_PASSWORD','LOGSTASH_PASSWORD','EDMIR_PASSWORD','KIBANA_ENCRYPTION_KEY')}
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write(''.join(k+'='+v+'\n' for k,v in values.items()))
    login=LOCAL/'LOGIN.txt'
    fd=os.open(login,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:f.write('Kibana: http://127.0.0.1:15601\nUser: edmir\nPassword: '+values['EDMIR_PASSWORD']+'\n仅限本地；不要提交 Git 或粘贴到聊天。\n')
    print('Local credentials created; passwords are in .local/LOGIN.txt')

def request(path,body=None,method=None,kibana=False):
    password=credentials()['ELASTIC_PASSWORD']
    auth=base64.b64encode(('elastic:'+password).encode()).decode()
    data=None if body is None else json.dumps(body).encode()
    headers={'Authorization':'Basic '+auth,'Content-Type':'application/json','kbn-xsrf':'edmir-local-setup'}
    req=urllib.request.Request(('http://127.0.0.1:15601' if kibana else 'http://127.0.0.1:19200')+path,data=data,headers=headers,method=method or ('POST' if data is not None else 'GET'))
    with urllib.request.urlopen(req,timeout=20) as response:return json.load(response)

def configure_es():
    c=credentials()
    request('/_security/user/kibana_system/_password',{'password':c['KIBANA_PASSWORD']})
    request('/_security/role/edmir_writer',{'cluster':['monitor'],'indices':[{'names':['edmir-ops-*'],'privileges':['write','create_index','view_index_metadata']}]},'PUT')
    request('/_security/user/edmir_writer',{'password':c['LOGSTASH_PASSWORD'],'roles':['edmir_writer']},'PUT')
    request('/_security/role/edmir_reader',{'cluster':[],'indices':[{'names':['edmir-ops-*'],'privileges':['read','view_index_metadata']}]},'PUT')
    request('/_security/user/edmir',{'password':c['EDMIR_PASSWORD'],'roles':['kibana_admin','edmir_reader'],'full_name':'EDM-IR 本地实验日志'},'PUT')
    keyword=lambda:{'type':'keyword','ignore_above':2048}
    props={'@timestamp':{'type':'date'},'message':{'type':'text'},'metadata':{'type':'flattened'},
      'event':{'properties':{**{k:keyword() for k in ('id','action','outcome','category','type','dataset','provider')},'created':{'type':'date'},'duration':{'type':'long'}}},
      'experiment':{'properties':{k:keyword() for k in ('id','stage','shape','candidate','kind')}},
      'evidence':{'properties':{k:keyword() for k in ('kind','timestamp_basis','source','sha256','record')}},
      'process':{'properties':{'exit_code':{'type':'integer'},'command_line':{'type':'text'},'pid':{'type':'long'}}},
      'log':{'properties':{'level':keyword(),'file':{'properties':{'path':keyword()}},'offset':{'type':'long'}}}}
    request('/_index_template/edmir-operations',{'index_patterns':['edmir-ops-*'],'priority':200,'template':{'settings':{'number_of_shards':1,'number_of_replicas':0,'index.mapping.total_fields.limit':1000},'mappings':{'dynamic':False,'properties':props}}},'PUT')
    print('Elasticsearch identities and index template configured')

def import_objects(objects):
    boundary='edmir'+secrets.token_hex(12)
    payload=(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in objects)).encode()
    body=('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="edmir.ndjson"\r\nContent-Type: application/ndjson\r\n\r\n').encode()+payload+('\r\n--'+boundary+'--\r\n').encode()
    auth=base64.b64encode(('elastic:'+credentials()['ELASTIC_PASSWORD']).encode()).decode()
    req=urllib.request.Request('http://127.0.0.1:15601/api/saved_objects/_import?overwrite=true',data=body,headers={'Authorization':'Basic '+auth,'kbn-xsrf':'edmir','Content-Type':'multipart/form-data; boundary='+boundary})
    with urllib.request.urlopen(req,timeout=45) as r:result=json.load(r)
    if not result.get('success'):raise RuntimeError(json.dumps(result))
    return result

def configure_kibana():
    request('/api/data_views/data_view',{'data_view':{'id':'edmir-operations','title':'edmir-ops-*','name':'EDM-IR 实验操作日志','timeFieldName':'@timestamp'},'override':True},kibana=True)
    objects=[]
    specs=[('all','实验操作时间线',''),('failures','失败与未完成','event.outcome: failure'),('recovery','数据回收与实例状态','event.action: (artifact_verified or recovery_verified or instance_state or platform_operation)'),('commands','命令与输出','event.action: (command_started or command_output or command_finished)')]
    for name,title,q in specs:
        objects.append({'type':'search','id':'edmir-'+name,'attributes':{'title':title,'description':'EDM-IR 操作记录；历史导入与实时记录分开标注。','columns':['experiment.id','event.action','event.outcome','message','evidence.kind'],'sort':[['@timestamp','desc']],'kibanaSavedObjectMeta':{'searchSourceJSON':json.dumps({'query':{'query':q,'language':'kuery'},'filter':[],'indexRefName':'kibanaSavedObjectMeta.searchSourceJSON.index'})}},'references':[{'type':'index-pattern','id':'edmir-operations','name':'kibanaSavedObjectMeta.searchSourceJSON.index'}]})
    text='''# EDM-IR · 实验操作中心\n\n按实验 ID 追踪准备、命令、采集、失败、回收与平台操作。所有时间显示为 Asia/Singapore。\n\n**实验已暂停。** 历史导入不是完整操作审计：命令未留痕的部分不会补造；文件时间不冒充实际操作时间。来源与时间依据在每条记录的 evidence 中。\n\n未来命令通过 `ops.py run` 记录；平台按钮操作通过 `ops.py record` 记录证据。停止进程、退出 SSH、容器关闭均不等于平台停费。\n\n[打开 Discover 搜索全部记录](/app/discover) · 查询示例：`event.outcome: failure`、`experiment.id: shadeform-w1-20261111`\n'''
    text=text.replace('**实验已暂停。** ', '各实验运行状态以对应日志和服务器状态为准。')
    objects.append({'type':'visualization','id':'edmir-intro','attributes':{'title':'EDM-IR 记录说明','visState':json.dumps({'title':'EDM-IR 记录说明','type':'markdown','params':{'markdown':text,'fontSize':12,'openLinksInNewTab':False},'aggs':[]}), 'uiStateJSON':'{}','kibanaSavedObjectMeta':{'searchSourceJSON':'{}'}},'references':[]})
    panels=[];refs=[]
    layouts=[('intro','visualization',0,0,48,11),('failures','search',0,11,24,18),('recovery','search',24,11,24,18),('all','search',0,29,48,24),('commands','search',0,53,48,20)]
    for i,(name,kind,x,y,w,h) in enumerate(layouts):
        ref='panel_'+str(i);refs.append({'name':ref,'type':kind,'id':'edmir-'+name})
        panels.append({'panelIndex':str(i),'type':kind,'gridData':{'x':x,'y':y,'w':w,'h':h,'i':str(i)},'panelRefName':ref,'embeddableConfig':{}})
    objects.append({'type':'dashboard','id':'edmir-operations-dashboard','attributes':{'title':'EDM-IR 实验操作中心','description':'命令、失败、数据回收与费用操作的可检索记录。','panelsJSON':json.dumps(panels),'optionsJSON':json.dumps({'useMargins':True,'hidePanelTitles':False}),'timeRestore':False,'version':1,'kibanaSavedObjectMeta':{'searchSourceJSON':json.dumps({'query':{'query':'','language':'kuery'},'filter':[]})}},'references':refs})
    (LOCAL/'saved-objects.ndjson').write_text(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in objects))
    print('Kibana imported:',import_objects(objects).get('successCount'))

def ensure_kibana():
    # Ordinary startup must not overwrite saved user layout/search changes.
    try:
        request('/api/saved_objects/dashboard/edmir-operations-dashboard',kibana=True)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        configure_kibana()
    else:
        print('Existing Kibana dashboard retained')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','es','kibana','kibana-ensure','status']);a=p.parse_args()
    if a.action=='init':init()
    elif a.action=='es':configure_es()
    elif a.action=='kibana':configure_kibana()
    elif a.action=='kibana-ensure':ensure_kibana()
    else:
        print(json.dumps({'elasticsearch':request('/_cluster/health'),'events':request('/edmir-ops-*/_count'),'kibana':request('/api/status',kibana=True).get('status')},indent=2))
if __name__=='__main__':main()
