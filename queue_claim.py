#!/usr/bin/env python3
"""큐 한 줄을 원자적으로 집거나(claim) 결과를 적는다(finish).

다른 서버가 SSH 로 이걸 호출해 같은 큐를 나눠 쓴다. **락이 걸리는 파일과 그 락을
잡는 프로세스가 모두 이 기계 위에 있으므로** NFS 락 문제 없이 원자성이 보장된다.

    ssh kyccj@canus python3 /home/kyccj/PycharmProjects/TensorFlow-SNNs/queue_claim.py \
        claim --tag gpu0@other
    -> "12\tr19c10\tl2\t4.5e-7\t1"   (없으면 빈 출력)

    ssh ... queue_claim.py finish 12 done
표준 라이브러리만 쓴다 — 원격에 conda 환경이 없어도 system python3 로 돈다.
"""
import argparse, fcntl, os, sys

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_queue/QUEUE.tsv')
COLS = 8


def _edit(path, fn):
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.read().split('\n')
            out = fn(lines)
            f.seek(0); f.write('\n'.join(lines)); f.truncate()
            return out
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def do_claim(path, tag):
    def fn(lines):
        cand = []
        for i, l in enumerate(lines):
            if not l or l.startswith('#'):
                continue
            c = l.split('\t')
            if len(c) >= COLS and c[0] == 'todo':
                cand.append((int(c[1]) if c[1].isdigit() else 9, i, c))
        if not cand:
            return None
        cand.sort(key=lambda t: (t[0], t[1]))
        _, i, c = cand[0]
        c[0] = f'run:{tag}'
        lines[i] = '\t'.join(c)
        return (i, c)
    return _edit(path, fn)


def do_finish(path, idx, status):
    """결과를 적되 **어느 서버가 돌렸는지 남긴다** (done:sejong:gpu5).

    canus 와 sejong 의 환경이 다르다 (TF 2.12.1/2.11.0, numpy 1.23.5/1.26.4,
    CUDA 11.8/11.2). 조건별로 어느 서버에서 몇 개가 돌았는지 모르면 서버 효과와
    방법 효과를 분리할 수 없다. 상태 줄에 출처를 보존한다."""
    def fn(lines):
        c = lines[idx].split('\t')
        who = c[0][4:] if c[0].startswith('run:') else ''
        c[0] = f'{status}:{who}' if who and status in ('done',) else status
        lines[idx] = '\t'.join(c)
        return True
    return _edit(path, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['claim', 'finish', 'status'])
    ap.add_argument('idx', nargs='?', type=int)
    ap.add_argument('state', nargs='?')
    ap.add_argument('--tag', default='unknown')
    ap.add_argument('--queue', default=Q)
    a = ap.parse_args()
    if a.action == 'claim':
        got = do_claim(a.queue, a.tag)
        if got:
            i, c = got
            print('\t'.join([str(i)] + c[2:6]))
    elif a.action == 'finish':
        do_finish(a.queue, a.idx, a.state)
    else:
        n = {}
        for l in open(a.queue):
            if l.startswith('#') or not l.strip():
                continue
            k = l.split('\t')[0].split(':')[0]
            n[k] = n.get(k, 0) + 1
        print('  '.join(f'{k}={v}' for k, v in sorted(n.items())))


if __name__ == '__main__':
    main()
