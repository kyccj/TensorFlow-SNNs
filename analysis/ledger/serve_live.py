"""실험 원장 로컬 실시간 보기 (canus).

1분마다 refresh.sh 로 두 서버를 다시 긁어 페이지를 만들고, 127.0.0.1 에만 연다.
외부에서는 안 보인다. PC 에서 터널로 본다:
    ssh -L 8765:127.0.0.1:8765 kyccj@161.122.23.138   ->  http://localhost:8765
노트 체크(artifact db)는 여기서는 안 보인다 — claude.ai 원장에서만.

쓰는 법:  nohup python analysis/ledger/serve_live.py > /tmp/eip_ledger_live/serve.log 2>&1 &
"""
import functools, http.server, os, subprocess, threading, time

ROOT = os.path.dirname(os.path.abspath(__file__))
W = '/tmp/eip_ledger_live'           # claude 세션 갱신(/tmp/eip_ledger 등)과 폴더를 나눈다
PORT = 8765
EVERY = 60


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def refresh_loop():
    while True:
        t0 = time.time()
        r = subprocess.run(['bash', os.path.join(ROOT, 'refresh.sh'), W],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(f'{W}/eip_ledger.html'):
            html = open(f'{W}/eip_ledger.html', encoding='utf-8').read()
            # 1분마다 새로 읽는다 (정렬·데이터셋 선택은 localStorage 라 유지된다)
            html = html.replace('<title>', f'<meta charset="utf-8"><meta http-equiv="refresh" content="{EVERY}"><title>', 1)
            tmp = f'{W}/index.html.tmp'
            open(tmp, 'w', encoding='utf-8').write(html)
            os.replace(tmp, f'{W}/index.html')          # 읽는 중에 반쪽 파일이 보이지 않게
            log(f'갱신 {time.time() - t0:.0f}s')
        else:
            log(f'갱신 실패 rc={r.returncode} — 이전 페이지 유지\n{r.stderr[-500:]}')
        time.sleep(max(5, EVERY - (time.time() - t0)))


class Quiet(http.server.SimpleHTTPRequestHandler):
    # 페이지에 charset 선언이 없어 한글이 깨졌다 (09-29). 헤더로도 UTF-8 을 알린다
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.html': 'text/html; charset=utf-8'}

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    os.makedirs(W, exist_ok=True)
    threading.Thread(target=refresh_loop, daemon=True).start()
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), functools.partial(Quiet, directory=W))
    log(f'http://127.0.0.1:{PORT}  (갱신 {EVERY}s, 작업폴더 {W})')
    srv.serve_forever()
