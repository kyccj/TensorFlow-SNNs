"""LaTeX 구조 검사 — pdflatex 이 없을 때의 대체 검증.
괄호/환경 균형, \cite 키의 refs.bib 존재, \input 대상 파일 존재를 본다.
컴파일을 대신하지 못한다 — TeX Live 설치 후 반드시 실제 빌드로 재검증한다."""
import re, sys, os

root = sys.argv[1] if len(sys.argv) > 1 else 'paper'
bib = os.path.join(root, 'refs.bib')
keys = set(re.findall(r'^@\w+\{([^,]+),', open(bib).read(), re.M)) if os.path.exists(bib) else set()

texs = []
for d, _, fs in os.walk(root):
    texs += [os.path.join(d, f) for f in fs if f.endswith('.tex')]

bad = 0
for t in sorted(texs):
    src = open(t, errors='replace').read()
    code = re.sub(r'(?<!\\)%.*', '', src)          # 주석 제거 (\% 는 남김)
    if code.count('{') != code.count('}'):
        print(f'{t}: 중괄호 불균형 {{={code.count("{")} }}={code.count("}")}'); bad += 1
    if code.count('$') % 2:
        print(f'{t}: $ 홀수개 ({code.count("$")})'); bad += 1
    begins = re.findall(r'\\begin\{(\w+\*?)\}', code)
    ends   = re.findall(r'\\end\{(\w+\*?)\}', code)
    for e in set(begins) | set(ends):
        if begins.count(e) != ends.count(e):
            print(f'{t}: 환경 {e} 불균형 begin={begins.count(e)} end={ends.count(e)}'); bad += 1
    for c in re.findall(r'\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}', code):
        for k in (x.strip() for x in c.split(',')):
            if k and k not in keys:
                print(f'{t}: refs.bib 에 없는 인용 키 `{k}`'); bad += 1
    for inp in re.findall(r'\\input\{([^}]+)\}', code):
        p = os.path.join(root, inp if inp.endswith('.tex') else inp + '.tex')
        if not os.path.exists(p):
            print(f'{t}: \\input 대상 없음 {p}'); bad += 1

print(f'\n검사한 .tex {len(texs)}개, refs.bib 키 {len(keys)}개, 문제 {bad}건')
print('주의: 이것은 컴파일이 아니다. TeX Live 설치 후 `make -C paper` 로 재검증할 것.')
sys.exit(1 if bad else 0)
