#!/usr/bin/env python
"""_queue/QUEUE.tsv 를 소비해 빈 GPU 에 실험을 채운다.

기존 run_paper.py 런처는 자기 GPU 집합을 고정으로 쥐고 있어서, 시드를 다 쓰면
GPU 가 놀아도 아무도 못 채웠다 (09-12 에 GPU 5 가 26시간 비었다). 이 실행기는
반대로 GPU 를 쥐지 않고 큐만 본다.

큐 파일은 탭 구분 TSV 이고 첫 열이 상태다:
    todo   아직 안 돌림
    run:<host>:<pid>   누가 집어갔는지
    done / fail:<rc>   결과

**claim 은 파일 락(fcntl.flock)으로 원자적이다.** 그래서 같은 큐 파일을 여러
프로세스 — 나아가 같은 파일시스템을 공유하는 여러 서버 — 가 동시에 봐도
한 항목을 두 번 돌리지 않는다.

쓰는 법:
    python queue_runner.py --gpus 3,4 [--queue _queue/QUEUE.tsv] [--dir _paper]
    python queue_runner.py --gpus 3,4 --dry     # 집어가기만 하고 실행 안 함
"""
import argparse, fcntl, os, shlex, socket, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = 8   # 상태 우선 조합 방법 손잡이 시드 예상착지 메모


def _rows(path):
    with open(path) as f:
        return [l.rstrip('\n') for l in f]


def claim(path, prio_first=True):
    """todo 한 줄을 원자적으로 집어 run: 으로 바꾸고 그 항목을 돌려준다."""
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.read().split('\n')
            cand = []
            for i, l in enumerate(lines):
                if not l or l.startswith('#'):
                    continue
                c = l.split('\t')
                if len(c) >= COLS and c[0] == 'todo':
                    cand.append((int(c[1]) if c[1].isdigit() else 9, i, c))
            if not cand:
                return None
            cand.sort(key=lambda t: (t[0], t[1]) if prio_first else (t[1],))
            _, i, c = cand[0]
            c[0] = f'run:{socket.gethostname()}:{os.getpid()}'
            lines[i] = '\t'.join(c)
            f.seek(0); f.write('\n'.join(lines)); f.truncate()
            return i, c
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def finish(path, idx, status):
    """queue_claim.do_finish 와 같은 규약 — done 이면 돌린 서버를 남긴다."""
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.read().split('\n')
            c = lines[idx].split('\t')
            who = c[0][4:] if c[0].startswith('run:') else ''
            c[0] = f'{status}:{who}' if who and status == 'done' else status
            lines[idx] = '\t'.join(c)
            f.seek(0); f.write('\n'.join(lines)); f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _ssh(host, port):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', str(port), host]


def remote_claim(host, remote_root, tag, port=22):
    """큐 주인 서버에서 한 줄을 집어온다. 락은 그쪽에서 걸리므로 원자적이다."""
    cmd = _ssh(host, port) + ['python3',
           shlex.quote(os.path.join(remote_root, 'queue_claim.py')),
           'claim', '--tag', shlex.quote(tag)]
    out = subprocess.check_output(cmd, timeout=120).decode().strip()
    if not out:
        return None
    p = out.split('\t')
    return int(p[0]), ['run'] + [''] + p[1:5] + ['', '']


def remote_finish(host, remote_root, idx, status, port=22):
    subprocess.run(_ssh(host, port) + ['python3',
                   shlex.quote(os.path.join(remote_root, 'queue_claim.py')),
                   'finish', str(idx), status], timeout=120)


def busy_gpus(max_mib=200):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'], timeout=60).decode()
    except Exception:
        return set(range(8))          # 못 읽으면 아무것도 안 건드린다
    b = set()
    for line in out.strip().split('\n'):
        i, m = [x.strip() for x in line.split(',')]
        if int(m) > max_mib:
            b.add(int(i))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='쓸 GPU (예: 3,4). 6,7 은 juyun 것이라 넣지 말 것')
    ap.add_argument('--queue', default=os.path.join(ROOT, '_queue/QUEUE.tsv'))
    ap.add_argument('--dir', default='_paper')
    ap.add_argument('--poll', type=int, default=180, help='빈 GPU 확인 주기(초)')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--remote', default='', metavar='USER@HOST',
                    help='큐 주인 서버. 주면 claim/finish 를 SSH 로 위임한다 '
                         '(공유 파일시스템 없이 서버 간 큐 공유)')
    ap.add_argument('--remote-root', default='/home/kyccj/PycharmProjects/TensorFlow-SNNs',
                    help='큐 주인 서버의 저장소 경로')
    ap.add_argument('--remote-port', type=int, default=23456,
                    help='큐 주인 서버의 sshd 포트 (canus 는 22 가 아니라 23456)')
    a = ap.parse_args()
    gpus = [int(x) for x in a.gpus.split(',')]
    # GPU 6,7 금지는 canus 에만 해당한다 (juyun 소유). 다른 서버는 자기 GPU 를 다 쓴다.
    if socket.gethostname() == 'canus':
        assert not (set(gpus) & {6, 7}), 'canus 의 GPU 6,7 은 juyun 소유다'
    running = {}                       # gpu -> (Popen, idx, item)

    while True:
        # 끝난 것 회수
        for g in list(running):
            p, idx, c = running[g]
            rc = p.poll()
            if rc is None:
                continue
            st = 'done' if rc == 0 else f'fail:{rc}'
            if a.remote:
                remote_finish(a.remote, a.remote_root, idx, st, a.remote_port)
            else:
                finish(a.queue, idx, st)
            print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {g}] DONE  "
                  f"{c[2]}-{c[3]}-{c[4]}-s{c[5]} rc={rc}", flush=True)
            del running[g]

        free = [g for g in gpus if g not in running and g not in busy_gpus()]
        got = None          # free 가 비면 아래 종료 판정에서 참조된다
        for g in free:
            tag = f'{socket.gethostname()}:gpu{g}'
            got = (remote_claim(a.remote, a.remote_root, tag, a.remote_port) if a.remote
                   else claim(a.queue))
            if not got:
                break
            idx, c = got
            _, _, combo, meth, knob, seed = c[:6]
            cmd = [sys.executable, '-u', os.path.join(ROOT, 'run_paper.py'),
                   combo, meth, knob, '--seeds', seed, '--gpus', str(g), '--dir', a.dir]
            print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {g}] START "
                  f"{combo}-{meth}-{knob}-s{seed}  (착지예상 {c[6]})", flush=True)
            if a.dry:
                (remote_finish(a.remote, a.remote_root, idx, 'todo', a.remote_port) if a.remote
                 else finish(a.queue, idx, 'todo'))
                print('   --dry: 되돌림, 실행 안 함', flush=True)
                continue
            running[g] = (subprocess.Popen(cmd, cwd=ROOT), idx, c)

        if not running:
            if a.remote:
                # 빈 GPU 가 있었는데도 못 집어왔으면 원격 큐가 비었다는 뜻이다.
                if free and got is None:
                    print('원격 큐 비었음 — 종료', flush=True)
                    return
            else:
                with open(a.queue) as f:
                    if not any(l.startswith('todo\t') for l in f):
                        print('큐 비었음 — 종료', flush=True)
                        return
        time.sleep(a.poll)


if __name__ == '__main__':
    main()
