"""Bounded-memory, resumable HTTP ranges; callers must verify the final digest."""
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import requests


def download(url, target, size, *, headers=None, workers=4, chunk_bytes=8*1024**2, progress=None):
    offset = target.stat().st_size if target.exists() else 0
    if offset > size:
        raise ValueError('partial exceeds declared size')

    local=threading.local()
    def part(span):
        begin, end = span
        if not hasattr(local,'session'):local.session=requests.Session()
        for attempt in range(4):
            try:
                started=time.monotonic()
                h = headers() if callable(headers) else dict(headers or {})
                h.update(Range=f'bytes={begin}-{end}', **{'Accept-Encoding': 'identity'})
                with local.session.get(url, headers=h, stream=True, timeout=(20, 40)) as response:
                    expected = f'bytes {begin}-{end}/{size}'
                    if response.status_code == 200 and begin == 0 and end == size-1:
                        pass
                    elif response.status_code != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError(f'range refused: HTTP {response.status_code}')
                    data = bytearray()
                    for block in response.iter_content(64*1024):
                        data.extend(block)
                        if time.monotonic()-started>120:
                            raise TimeoutError('range transfer exceeded 120 seconds')
                        if len(data) > end-begin+1:
                            raise ValueError('response exceeded requested range')
                    if len(data) != end-begin+1:
                        raise ValueError('truncated range')
                    if progress:progress({'offset':begin,'bytes':len(data),'seconds':round(time.monotonic()-started,3)})
                    return data
            except Exception as exc:
                if progress:progress({'offset':begin,'attempt':attempt+1,'error_type':type(exc).__name__})
                if attempt == 3:
                    raise
                time.sleep(1+attempt)

    with target.open('ab') as output, ThreadPoolExecutor(max_workers=workers) as pool:
        while offset < size:
            spans = [(begin, min(begin+chunk_bytes, size)-1)
                     for begin in range(offset, min(size, offset+workers*chunk_bytes), chunk_bytes)]
            # Only this bounded window is in memory. Append in byte order;
            # interrupted downloads resume exactly after the last durable write.
            for data in pool.map(part, spans):
                output.write(data)
                output.flush()
                offset += len(data)
