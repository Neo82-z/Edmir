import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
m=json.loads(Path(sys.argv[2]).read_text())
files=m.get("files",m)
errors=[]
for name,expected in files.items():
 f=(root/name).resolve()
 if not f.is_relative_to(root) or not f.is_file():
  errors.append((name,"missing/unsafe"));continue
 if f.stat().st_size!=expected["bytes"] or hashlib.file_digest(f.open("rb"),"sha256").hexdigest()!=expected["sha256"]:
  errors.append((name,"hash/size mismatch"))
print(json.dumps({"files":len(files),"errors":errors}))
raise SystemExit(bool(errors))
