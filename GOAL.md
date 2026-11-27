# 커널 최적화 GOAL

**H100에서 LLM 추론에 쓰이는 커널을 하나씩 개선하고, 다음 실험은 직전 실측 결과로 결정한다.** 한 실험이 끝나면 결과를 공개하고 후속 가설을 하나 선택한다. 연구 주제에는 종료 횟수를 두지 않는다. GPU 사용은 저장·재개할 수 있는 작은 작업으로 나눈다.

## 현재 인계 상태

| 실험 | 상태 | 근거와 결정 |
| --- | --- | --- |
| RMSNorm 워프 수 | 완료 | 1×4096에서 8워프가 컴파일된 PyTorch보다 약 14% 빨랐다. 1024×8192에서의 비교를 후속 실험으로 선택했다. [첫 측정](results/h100.json) · [재측정](results/h100-repeat.json) |
| 프로그램당 1·2·4행 처리 | 완료 | 정확도 60건을 통과했다. 1024×8192에서는 새 설정이 두 기준을 모두 5% 앞서지 못해 기존 1행 구현을 유지했다. [실험 보고서](results/row-tiles/REPORT.md) |
| 은닉 차원에 따른 2행 타일의 손익 | 완료 | 1024×4104에서만 2행·4워프가 두 시드에서 두 기준을 5% 이상 앞섰다. 이 입력에 해당 설정을 권장한다. [결과와 원시 기록](results/width-sweep/REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/2) |
| 프리필 BF16 행렬곱의 지속 처리량 | 완료 | 20구간에서 총 902.18초를 측정했다. 16,384행의 처리량 중앙값은 682.19 TFLOP/s였다. 사용률 표본이 충분한 구간은 8개였고, 다섯 시드 중 네 개에서 지속 부하를 확인한 입력 크기는 없었다. [결과와 자원 사용 그래프](experiments/prefill-gemm/REPORT.md) |
| 프리필 중 GPU 사용률의 관측 공백 | 완료 | 동일한 H100과 프리필 입력에서 직접 NVML 계측이 6/6 시드, CLI 계측이 3/6 시드에서 유효 표본과 최대 공백 기준을 통과했다. [최종 보고서](https://github.com/hyscodebase/h100-rmsnorm-lab/blob/codex/h100-observer-coverage/experiments/observer/REPORT.md) |
| 잔차 덧셈과 RMSNorm 융합 | 완료 | 12/12회에서 Triton은 두 출력의 정확도 검사를 통과했다. 목표 세 시드에서 10.068–10.113 µs였고, 컴파일 비교군 48건은 분모 반올림 검사에서 탈락했다. [실험 보고서](experiments/rmsnorm-residual/REPORT.md) |
| 컴파일 중간값의 정밀도 설정 | 완료 | 새 시드와 독립 프로세스로 설정이 실제 켜지고 꺼졌음을 확인했다. 두 경우 모두 같은 분모 검사 불일치가 재현됐다. 시드 105에서 BF16 저장 경계를 명시한 2단계 컴파일 기준과 비트 반올림을 명시한 단일 그래프의 정확도를 먼저 대조한다. [다음 실험 계획](experiments/rmsnorm-residual/materialized/PLAN.md) [원시 기록](experiments/rmsnorm-residual/precision/checkpoint.json) |
| 정확도가 일치하는 컴파일 기준선과 비교 | 완료 | 새 컴파일 기준 두 종류가 정확도 검사를 통과했다. 1024×4096에서 Triton은 가장 빠른 컴파일 모드보다 세 시드에서 각각 23.44%, 23.83%, 23.56% 낮은 지연시간으로 5% 기준을 통과했다. [결과와 재현 명령](experiments/rmsnorm-residual/materialized/REPORT.md) |
| vLLM과 동일한 제자리 연산 비교 | 정확도 단계 완료 | 공식 vLLM 커널과 Triton이 상쇄 입력 419만 개를 비트 단위로 동일하게 출력했다. FP64 사전 기준은 BF16 반올림 경계의 10개 원소에서 두 구현 모두 충족하지 못해 속도 측정을 보류했다. [재현 데이터와 다음 검증](experiments/rmsnorm-residual/vllm/REPORT.md) |
| vLLM의 FP32 누산과 BF16 경계 검증 | 완료 | 새 시드 113은 통과했다. 114–116에서는 허용 범위 끝의 인접 BF16 값에서 구현 간 비트가 갈려 각 시드의 판정 기준을 통과하지 못했다. 시드 114의 탈락 47개를 재현했다. [원시 결과](experiments/rmsnorm-residual/vllm/REPORT.md) |
| vLLM에 맞춘 FP32 합산 순서 | 완료 | 이전 47개 차이를 해소했다. 다섯 시드에서 비교한 약 5.70억 개의 결과가 비트 단위로 일치했다. 세 타이밍 시드에서 지연시간은 vLLM보다 7.37~7.82% 높았다. [보고서](experiments/rmsnorm-residual/vllm/GROUPED_REPORT.md) |
| 동일한 합산 순서의 워프 셔플 구현 | 완료 | 비트 일치를 유지하면서 공유 메모리 읽기 명령이 128개에서 2개로 줄었다. vLLM 대비 세 시드에서 지연시간이 7.43~7.58% 줄어 5% 기준을 통과했다. [결과와 그래프](experiments/rmsnorm-residual/vllm/SHUFFLE_REPORT.md) |
| 256행 경계와 다른 크기에서 재검증 | 완료 | 256·512·2048행에서 비트가 일치했다. 256행은 12.02~12.18%, 512행은 10.38~10.51% 빨랐다. 2048행은 4.892~4.997%였다. 128행은 모든 타이밍 시드에서 정확도 기준에 탈락했다. [실험 보고서와 원시 기록](experiments/rmsnorm-residual/vllm/TRANSFER_REPORT.md) |
| 128행에서 vLLM의 FP32 합산 순서 재현 | 완료 | 예전 비트 차이 238개를 재현한 뒤 모두 해소했다. 일곱 체크포인트에서 약 1.47억 개 결과가 비트 단위로 일치했다. 지연시간은 1.044~1.147% 줄어 5% 기준에는 못 미쳤다. [결과와 원시 기록](experiments/rmsnorm-residual/vllm/SMALL_ROWS_REPORT.md) |
| 128행에서 물리 워프 16개에 벡터 분산 | 완료 | 세 독립 시드에서 출력 비트를 맞춘 상태로 지연시간이 vLLM 대비 15.834~15.985%, 기존 8워프 대비 15.010~15.158% 줄었다. 컴파일러가 생성한 중복 합산을 고정 레이아웃으로 해소했다. [세 구현의 실측과 재현](experiments/rmsnorm-residual/vllm/SMALL_ROWS_16W_REPORT.md) |
| 128행을 8개 CUDA 스트림에서 연속 처리 | 완료 | 162.355초에 65.72억 행을 반복 처리했다. 16워프 처리량은 vLLM 대비 6.718~7.405% 높았고 8워프와는 −0.578~+0.040% 차이였다. NVML 사용률 중앙값은 9개 조합 모두 100%였고, 7개가 관측 범위 기준을 통과했다. [실측과 자원 기록](experiments/rmsnorm-residual/vllm/SUSTAINED_128_REPORT.md) |
| 8개 스트림의 입력 복사와 커널 실행 시간 분리 | 완료 | 연산만 계측해도 16워프는 8워프 대비 0.003~0.103% 높아 5% 기준에는 못 미쳤다. Triton의 복사는 분리된 CUDA 시간의 52.23~52.46%였다. [원시 이벤트와 실측](experiments/rmsnorm-residual/vllm/COPY_SPLIT_REPORT.md) |
| 원본을 보존하며 출력 버퍼에 직접 쓰기 | 완료 | 8스트림에서 두 출력의 비트를 맞추고 원본을 보존했다. 직접 출력 처리량은 세 시드에서 초당 8,557만~8,646만 행으로 가장 빠른 기준선의 2.059~2.080배였다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/SUSTAINED_OUT_OF_PLACE_REPORT.md) |
| 복사 없는 두 커널과 융합 커널 비교 | 완료 | 128×4096에서 두 출력의 비트를 맞추고 원본을 보존했다. 융합은 두 커널 기준보다 세 독립 시드에서 15.125%, 21.429%, 16.547% 높은 처리량을 기록해 5%를 모두 넘었다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/COPY_FREE_FUSION_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/29) |
| 255행에서 복사 없는 융합 이득 재검증 | 완료 | 128행과 같은 바이너리로 두 출력을 비트 단위로 맞췄다. 융합 처리량은 복사 없는 두 커널보다 세 시드에서 각각 12.766%, 14.026%, 12.809% 높았다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/COPY_FREE_ROWS_255_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/31) |
| 256행에서 vLLM의 합산 순서가 바뀐 뒤 비교 | 완료 | 기존 16워프는 정규화 결과 1,161개에서 비트가 달랐다. 새 8워프의 두 출력은 모두 일치했고, 융합 처리량은 복사 없는 두 커널보다 세 시드에서 12.381~12.485% 높았다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/COPY_FREE_ROWS_256_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/33) |
| 512행에서 8워프 융합 이득 재검증 | 완료 | 256행과 같은 바이너리로 두 출력을 비트 단위로 맞췄다. 복사 없는 두 커널 대비 융합 처리량은 독립 시드 세 개에서 19.741~19.759% 높았다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/COPY_FREE_ROWS_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/35) |
| 512행에서 DRAM과 L2 트래픽 비교 | 호스트 권한 대기 | 시드 165에서 Nsight Compute의 범위 계측이 `ERR_NVGPUCTRPERM`으로 중단됐다. 두 시도의 원시 오류와 자원 표본을 보존했다. [검증 기록](experiments/rmsnorm-residual/vllm/COPY_FREE_MEMORY_512_QUALIFICATION.md) · [측정 이슈](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/36) |
| 512행에서 잔차 생성과 정규화 사이의 실행 간격 비교 | 완료 | 같은 64개 커널을 쌍마다 이어 실행하면 덧셈을 먼저 모아서 실행할 때보다 세 시드에서 3.176~3.262% 높았다. 두 출력은 일치했고 5% 기준에는 못 미쳤다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/SPLIT_SCHEDULE_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/39) |
| 512행의 복사 없는 덧셈 타일 크기 최적화 | 완료 | 8,192원소 타일은 세 독립 시드에서 2,048원소보다 0.438~0.453% 빨라 5% 채택 기준에는 못 미쳤다. 융합은 더 빠른 분리 구현보다 19.206~19.230% 높은 처리량을 보였으며 두 BF16 출력이 일치했다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/SPLIT_ADD_TILES_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/41) |
| 512행에서 두 BF16 프로젝션 사이의 융합 이득 | 완료 | 융합으로 체인 처리량이 1.256~1.359% 높아 세 독립 시드에서 1% 기준을 통과했다. 두 프로젝션과 잔차·정규화 출력의 비트가 모두 일치했다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/PROJECTION_CHAIN_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/43) |
| 두 번째 BF16 프로젝션의 너비를 늘렸을 때 융합 이득 | 완료 | 12,288열에서 세 독립 시드의 융합 이득이 0.362~0.536%로 1% 기준에 못 미쳤다. 네 출력의 비트가 모두 일치했다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/PROJECTION_TAIL_12288_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/45) |
| 두 번째 BF16 프로젝션이 6,144열일 때 융합 이득 | 완료 | 체인 처리량이 0.911~1.034% 높았다. 네 출력은 모든 시드에서 일치했고 한 시드만 1%를 넘었다. [실측·자원 기록](experiments/rmsnorm-residual/vllm/PROJECTION_TAIL_6144_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/47) |
| 기존 합산 순서를 유지한 4개 물리 워프 | 완료 | 두 BF16 출력이 일치했다. 세 독립 시드에서 처리량은 0.032~0.090% 낮았고, 관측 전력 중앙값은 10.07~11.56W 낮았다. [실측·컴파일·자원 기록](experiments/rmsnorm-residual/vllm/FOUR_PHYSICAL_WARPS_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/49) |
| 같은 수의 행을 처리할 때 4워프와 8워프의 에너지 비교 | 완료 | 누적 카운터 구간의 행당 에너지는 0.844~2.183% 줄었다. 두 시드에서만 1.5%를 넘었고 조회 경계 여유가 최대 2.274초였다. [측정·민감도 분석](experiments/rmsnorm-residual/vllm/FOUR_WARP_ENERGY_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/51) |
| 에너지 카운터 경계를 50ms 이내로 제한 | 완료 | 두 완전한 시드에서 행당 에너지가 2.237~2.244% 줄었다. 시드 199는 조회 지연으로 한 쌍만 채택됐다. [실측·지연 분석](experiments/rmsnorm-residual/vllm/ENERGY_COUNTER_BOUNDARIES_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/53) |
| NVML 표본 시각으로 에너지 작업 구간 검증 | 완료 | 시작 조회 지연이 있었던 세 구간의 표본 경계가 8ms 이내였다. 두 순서가 모두 유효한 시드에서 행당 에너지는 1.360~1.779% 줄었다. 한 시드만 1.5% 기준을 넘었고, 시드 202는 한 쌍만 채택됐다. [실측과 자원 기록](experiments/rmsnorm-residual/vllm/TIMESTAMPED_NVML_ENERGY_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/55) |
| 고정 작업량을 8배 늘려 에너지 경계 오차 평가 | 완료 | 세 독립 시드에서 두 BF16 출력이 일치했고 여섯 비교쌍이 유효했다. 행당 에너지는 관측값 기준 1.180~1.599% 줄었다. 1,000W 가정의 절감률 하한이 1.5%를 넘긴 시드는 207 하나였다. [보고서와 체크포인트](experiments/rmsnorm-residual/vllm/LONG_WINDOW_ENERGY_512_REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/57) |
| 512행 BF16 융합 커널의 입력 로드에서 L1 우회 | 완료 | 세 독립 시드에서 `.cg` 처리량 이득은 0.000779~0.007549%로 5% 채택 기준에 못 미쳤다. 두 출력은 비트 단위로 일치했고 H100 자원 표본 133건을 보관했다. 8워프 기준선을 유지한다. [최종 보고서와 원시 기록](experiments/rmsnorm-residual/vllm/input-load-cache-512-results/REPORT.md) · [리뷰 PR](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/59) |
| RMSNorm 4104 인접 너비 재검증 | 완료 | 4104는 두 기준보다 5.33~5.56% 빨라 새 세 시드에서 권고를 재확인했다. 4103·4105도 각각 세 시드에서 기준을 넘었지만 기본 권고를 넓히지 않았고, 4112는 1행 대비 4.83·4.83·5.04%여서 세 시드 기준을 통과하지 못했다. 두 수치 게이트가 12개 checkpoint에서 모두 통과했다. [보고서·원시 기록](results/width-neighborhood/REPORT.md) · [PR #61](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/61) |
| 처리량 크기에서 RMSNorm 행 타일 검증 | 완료 | 32,768행의 폭 네 개와 시드 47–49, 총 12 checkpoint에서 양쪽 수치 게이트·메모리 검사가 통과했다. 2행·4워프 후보는 가장 빠른 1행 기준보다 폭 4103에서 2.986–3.083%, 4105에서 1.866–1.895% 빨랐고, 4104·4112에서는 1.156–1.220% 느렸다. 모두 사전 등록한 5% 채택선을 넘지 못했다. [보고서와 원시 기록](results/throughput-neighborhood/REPORT.md) · [PR #63](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/63) |
| 처리량 크기에서 4행 RMSNorm 타일 조사 | 완료 | 32,768행, 폭 4103·4105에서 4행 후보가 세 시드 모두 5% 기준을 통과했다. 폭 4104·4112에서는 약 1.1% 느려 전체 기본 설정 변경은 거부했다. [전체 12 checkpoint 실측·자원 감사](results/sustained-throughput/PROGRESS.md) · [PR #65](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/65) |
| 32,768행 인접 폭의 제한된 홀수·짝수 타일 선택 조사 | 진행 중 · 전체 매트릭스 해제 | 폭 8개·시드 53–55, 총 24 checkpoint를 계획했다. 이전 핀의 9개 checkpoint(19.83 CUDA-event 분)는 감사/복구 결함이 확인되어 archive에 보존하고 새 소스 분석에서 제외했다. active source `d42481a`에서 seed 53의 4,102·4,106 폭 2/24 완료, CUDA-event 4.41분, telemetry 878행이다. 두 짝수 폭 모두 4행 후보가 가장 빠른 1행 기준보다 1.62%, 1.55% 느려 5% gate에 미달했다. 후보는 `torch.compile`보다 각각 6.78%, 6.65% 높았다. GPU busy 중앙값/피크 100%, VRAM peak 5,064 MiB, 전력 중앙값/피크 634.1/699.8 W, 온도 peak 53 C였으나 query gap이 4.548초로 1.5초 기준을 넘어 8개 구간 모두 활동 telemetry 부적격이다. raw 표본은 진단용으로 보존하고 계산 증거는 유효하다. 첫 체크포인트 전 두 번의 SSH 전송 실패는 remote benchmark 전 단계여서 H100 시간을 쓰지 않았고, 총 네 영수증 중 두 번이 성공했다. PR #67 검토에는 blocker가 없어 전체 매트릭스 실행을 해제했다. 다음 단위는 `s53-w4107`; 최종 dispatch 결론은 아직 없다. [이슈 #66](https://github.com/hyscodebase/h100-rmsnorm-lab/issues/66) · [PR #67 리뷰](https://github.com/hyscodebase/h100-rmsnorm-lab/pull/67) · [연구 누적 PDF](output/pdf/H100_RMSNorm_연구누적기록_2026-11-27.pdf) · [PDF 생성기](scripts/build_h100_research_pdf.py) · [원자료 감사](results/width-dispatch/raw/partial-audit.json) · [첫 체크포인트 검토](results/width-dispatch/reviews/20261127-pilot-checkpoint-review.md) · [이전 핀 archive](results/width-dispatch/raw/archives/review-blockers-5-20261127/ARCHIVE_NOTE.md) · [수정 코드 검토](results/width-dispatch/reviews/20261127-runner-review-followup.md) |

## 다음 가설을 고르는 규칙

1. 하나의 입력 크기, 가장 경쟁력 있는 기준 구현, 바꿀 요소 하나를 정한다. 예상 이득, 수치 정확도, 채택 기준을 먼저 기록한다.
2. 커널이나 실행 환경을 바꾸면 새 실험으로 구분한다. 같은 커널 소스와 방법으로 후보와 기준을 함께 측정한다.
3. 정확도와 원시 지연시간을 보관한다. RMSNorm 실험에서 기본 설정을 바꾸려면 동일 후보가 두 시드 모두에서 최선의 기존 1행 설정과 컴파일된 PyTorch보다 각각 5% 이상 빨라야 한다.
4. 이득이 재현되면 인접 입력 크기와 실제 라이브러리 구현에서 검증한다. 이득이 사라지면 경계 입력이나 프로파일러 지표로 원인 후보 하나를 좁힌다. 새 가설은 직전 결과 파일과 연결한다.
5. 한 사이클이 끝나면 이 표에 결정과 근거를 추가하고 다음 계획 JSON을 만든다. 결과가 있는 개인 저장소에서 계속 진행한다. 상류 PR을 제안하려면 현재 상류 구현과의 비교 및 기여 범위를 따로 확인한다.

## 측정 근거 보관

완료한 구간마다 코드 버전과 해시, 재현 인수, 시작·종료 시각, 정확도 검사, 워밍업, 배치별 원시 시간과 GPU 자원 표본을 보관한다. 새로 수집하는 항목은 최초 관측 시점과 단위를 표시한다. 센서 누락과 재시도도 기록한다. 보조 계측값은 같은 장치·실행 시간에 맞춰 별도로 공개하고, 사전에 정한 성능 판정에는 원래 계측 절차를 적용한다. [프리필 실험의 기록 목록](experiments/prefill-gemm/evidence/README.md)

## GitHub에 남기는 실험 기록

실험마다 개인 저장소의 이슈에 질문, 비교할 기준 구현, 채택 조건을 적는다. 연결된 PR에는 측정 전에 정한 계획, 재현 명령, 시드별 원시 측정값, 결과 해석을 모은다. 구현 변경, 실측 결과, 리뷰에 따른 수정은 각각 내용을 설명하는 커밋으로 남긴다.

PR 리뷰에서는 정확도 검사 범위, 같은 실행에서 측정한 기준선, 단위와 통계값, 원시 파일에서 결론을 재현할 수 있는지를 확인한다. 지적 사항에는 수정 커밋과 재검증 결과를 연결한다. 머지 전에는 마지막 코드 변경까지 검토했는지 확인한다. 일시정지 중에는 다음 GPU 작업을 시작하지 않고, 재개할 때 현재 PR의 미해결 리뷰도 확인한다.

PR 리뷰와 H100 측정은 서로 독립적으로 진행할 수 있으므로, 리뷰가 대기 중인 동안 다음 체크포인트 한 개를 실행하고 끝나는 즉시 기록한다. H100에는 동시에 체크포인트 하나만 제출해 기준선과 후보 간 간섭 및 중복 실행을 막는다. 리뷰에서 실행 정확도·계측·결과 해석을 바꾸는 문제가 나오면 현재 체크포인트 경계에서 멈추고, 수정된 소스를 다시 검토한 뒤 이어간다.

## 끊어서 실행하기

현재 실험의 첫 실행에서 `python3 lab.py init --plan plans/prefill-gemm.json`으로 상태를 만든다. 같은 체크아웃에서 다음 명령을 사용한다.

```bash
python3 lab.py status
python3 lab.py step
python3 lab.py pause
python3 lab.py resume
python3 lab.py run --max-steps 2
```

- **짧게 작업할 때** `step`은 정확도 검사 한 묶음 또는 입력 크기 하나와 시드 하나의 측정만 실행한다. 프리필 계획은 20개 구간으로 나뉘고, 각 구간 안에서 수치 검증도 수행한다. RMSNorm 계획에서는 정확도 검사가 독립 구간이다.
- **이동하기 전** 다른 터미널에서 `pause`를 실행하면 진행 중인 측정이 끝난 뒤 다음 측정을 시작하지 않는다. `status`에서 `pausing`은 마무리 중, `paused`는 연결을 끊어도 되는 상태다. 노트북이 먼저 잠들거나 연결이 끊기면 다음 실행에서 결과를 저장하지 못한 한 구간을 다시 돌린다.
- **돌아왔을 때** `status`로 저장 지점을 확인하고 `resume` 후 `step` 또는 `run --max-steps N`으로 이어간다. 상태 조회와 일시정지는 오프라인에서도 가능하다.
- **Codex에서** “이동할게, 커널 GOAL 일시정지”라고 하면 측정 대기열과 채팅 GOAL을 함께 멈춘다. 작업을 재개할 때는 저장 지점과 소스 버전을 먼저 확인한다.
- **보관 위치** 기본 상태는 이 체크아웃의 `.lab/current`에 남는다. 다른 기기에서 이어갈 때는 같은 소스 버전과 상태 폴더를 함께 옮긴다. 기본 경로를 계속 쓸 때는 완료한 실험의 상태 폴더를 `.lab/history/<실험 ID>`에 보존한 다음 새 계획으로 초기화한다. 별도로 진행 중인 계획은 `--state-dir`로 선택한다.
