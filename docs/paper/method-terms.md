# Method 용어 대응표 (Task 5, 2026-09-14)

Method 절이 확정/변경되어도 §1·§2·§4·§5 등 다른 절 프로즈를 고치지 않기 위한 간접층이다.
§4.3 어블레이션 표(T3)를 포함해 **Method 밖에서 쓰는 행/조건 이름은 반드시 이 표의
오른쪽 두 열(논문 표기 / 본문 서술)만 쓴다.** 코드 플래그(왼쪽 열)를 §1·§2·§4 프로즈에
그대로 노출하지 않는다 — 노출해야 하면 `[내부용, 본문 반입 금지]` 로 표시하고 최종 원고
전에 제거한다.

Method 가 바뀌면 **이 파일만** 고친다. `paper/sections/*.tex` 는 건드리지 않는다.

| 코드 플래그 (`run_paper.py`) | 논문 표기 | 본문에서 부르는 법 |
|---|---|---|
| `prop` (전체) | Ours | the proposed regulariser |
| `abl_novmem` | w/o potential | the sub-threshold term |
| `abl_noinv` | w/o inversion | the inversion |
| `abl_nofinal` | w/o final-step | the single-step application |
| `abl_nolr` | fixed $\lambda$ | the ratio controller |
| `l2` | Plain L2 | the layer-wise $L_2$ penalty |
| `sm` | 1-softmax | the softmax weighting |
| `base` | No regularisation | the unregularised baseline |

## 표기 규약 (스펙 §6 — 사용자 지시)

- `1-maxnorm` 과 `maxnorm` 은 서로 다른 방법이다. 이 표에는 없지만 §2/§4 부록에서
  두 이름이 함께 나오면 반드시 `1−` 접두를 구분해서 쓴다 (예: 이 문서의 `sm` = `1-softmax`
  이지 `softmax` 가 아니다).
- 아직 이 표에 없는 조건이 추가되면(예: G4 완료 시 `l2 + loss-ratio` 대조군) 이 표에
  행을 먼저 추가한 뒤에 본문에서 부른다. 본문에서 코드 플래그를 먼저 쓰고 나중에
  이 표를 맞추지 않는다.

## 아직 비어 있는 자리

- G4 대조군(`l2` + loss-ratio 결합)이 채워지면 코드 플래그를 확인해 행을 추가한다.
  §2.4/§1 P6 CONTRIB-2 프로즈가 이 대조군을 `[내부용, 본문 반입 금지] l2 + loss-ratio`
  로만 지칭하고 있으므로, 확정되면 이 표의 "본문 서술" 열 표현으로 교체한다.
