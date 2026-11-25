import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

EPSILON = 1e-9
DEFAULT_REPEATS = 10
UNDECIDED = "UNDECIDED"

FILTER_LABEL_MAP = {
    "cross": "Cross",
    "x": "X",
}

EXPECTED_LABEL_MAP = {
    "+": "Cross",
    "cross": "Cross",
    "x": "X",
}


class Matrix2D:
    """N x N 2차원 패턴 및 필터 저장용 데이터 구조."""

    def __init__(self, size: int, data: Optional[List[List[float]]] = None) -> None: # size만 넘겨 호출하면 data는 빈 행렬(리스트), =None는 data파라미터 기본값, Optional은 실수형 2차원 리스트일 수 도있고 None일수도 있다는 의미의 내장 함수,
        if not isinstance(size, int) or size <= 0: # 내장 함수 isinstance => 행렬의 사이즈(N인 정방행렬이면 사이즈 N)가 int이면 T 아니면 F, 즉 사이즈가 정수가 아니거나 사이즈가 0이하인 분기
            raise ValueError(f"행렬 크기는 양의 정수여야 합니다: {size}") # 예외 강제 발생(raise는 예외를 강제 발생시키는 내장 함수)
        self.size = size
        if data is None: # 행렬(이중 리스트)자체가 None라면
            # : List[List[float]] 부분은 변수 타입 힌트(실제 코드에선 무시됨)
            self.data: List[List[float]] = [
                [0.0 for _ in range(size)] for _ in range(size) # [리스트에_넣을_값 for 변수 in 반복할_대상] 즉 하나의 행(리스트) 별로 0.0을 size 길이만큼 넣고 또 그렇게 만들어진 하나의 행(리스트)을 다시 큰 리스트에 size의 길이 만큼 넣음
            ] # 결국 size * size의 내부 데이터가 모두 0.0인 정방행렬(리스트 내부 리스트 형태) 생성 => self.size에 보관
        else: # any 함수는 리스트 or 제너레이터(여기선 제너레이터)등을 받아서 그 값중 하나라도 T면 T, 전부 F일때만 F를 반환 
            if len(data) != size or any((len(row) != size) for row in data): # 이중 리스트 중 바깥 리스트의 길이가 size와 같지 않거나(즉 열의 크기가 size와 같지 않거나), 내부 리스트의 크기가 size와 같지 않으면(즉 행의 크기가 size와 같지 않거나) => 정방 행렬인지 판단하는 분기임
                raise ValueError(f"{size}x{size} 정방행렬 형식이 아닙니다.")
            self.data = [[float(val) for val in row] for row in data] # 정방행렬이면 행렬의 모든 값을 float로 형변환하여 self.data에 저장

    def get(self, row: int, col: int) -> float: 
        # 행렬 내부 특정 좌표(row, col)의 값을 읽어오는 멤버 함수.
        self._check_bounds(row, col) # 조회하려는 행과 열이 out of index에 걸리진 않는지 확인하는 함수
        return self.data[row][col] # out of index에 안걸리면 행렬 좌표에 해당하는 값 반환 

    def set(self, row: int, col: int, value: float) -> None:
        # 특정 위치 (row, col)에 값을 저장하는 함수.
        self._check_bounds(row, col)
        num = float(value)
        if not math.isfinite(num): # 특정 좌표에 저장될 실수가 무한대(inf) 혹은 NaN(무한대 간 연산결과 등)은 아닌지 확인하는 분기
            raise ValueError(f"유효하지 않은 숫자입니다: {value}")
        self.data[row][col] = num # num이 유한소수이면 특정 좌표 값에 num 대입

    def _check_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self.size and 0 <= col < self.size): # size*size 정방행렬에서 행과 열이 0보다 작고 size보다 크거나 같은지 확인하는 분기(행, 열은 결국 리스트내부 인덱스이기 때문에 0행 0열부터 시작 즉 size보다 작은것이 타당)
            raise IndexError(
                f"인덱스 범위 초과: ({row}, {col}), 크기: {self.size}x{self.size}"
            )

    def to_flat(self) -> List[float]: # 반환값 = 1차원 리스트
        # 2차원 리스트를 길이 N^2의 1차원 리스트로 변환하는 함수(최적화용)(EX : 3*3 정방행렬이면 길이가 9인 리스트로 변환).
        flat: List[float] = []
        for row in self.data: # 각 행
            for val in row: # 각 행별 값
                flat.append(val) # 그 값을 list에 추가
        return flat # 그 리스트 반환

    def format_grid(self) -> str:
        # 콘솔 확인용 문자열 표현을 반환하는 함수.
        lines: List[str] = []
        for row in self.data:
            lines.append("  " + " ".join(f"{v:g}" for v in row)) # join은 행렬의 모든 값을 " "구분자로 하여 문자열로 출력을 돕는 함수임(EX : ",".join(["0", "1", "0"]) => "0,1,0"), :g는 v가 소수점 하위가 모두 0이면 정수형으로 아니라면 불필요한 0을제외하고 깔끔한 소수로 반환해주는 역할(포맷지정자, EX : 1.0 => 1, 1.50 => 1.5)
        return "\n".join(lines) # 출력도 문자열로(구분자는 \n)

    @classmethod # 객체를 미리 만들지 않고, 클래스 이름으로 바로 호출할 수 있는 메서드라는 의미 => 따라서 self가 없음
    def from_list(cls, raw: Any, expected_size: Optional[int] = None ) -> "Matrix2D": # Any는 어떤 타입이던 다 들어올수 있는 타입(아르세우스 레전드 플레이트같은거임) ㅋㅋ
        # 2차원 리스트의 스키마/크기/숫자 유효성을 검증하고 Matrix2D를 생성 하는 함수. (여기서 스키마는 약속된 설계도인지 즉 이중 리스트인지 여부같은걸 의미) 
        if not isinstance(raw, list) or len(raw) == 0: # raw가 리스트가 아니거나 길이가 0이면
            raise ValueError("2차원 배열 형식이 아니거나 비어 있습니다.")

        row_count = len(raw)
        target_size = expected_size if expected_size is not None else row_count # 삼항 연산자(참일_때_값 if 조건 else 거짓일_때_값)

        if row_count != target_size: # raw(행렬)가 target_size * target_size 정방행렬인지 보는 분기(그 중 행)
            raise ValueError(
                f"행 수 불일치: 기대 {target_size}행, 실제 {row_count}행"
            )

        parsed: List[List[float]] = []
        for r_idx, row in enumerate(raw):
            if not isinstance(row, list):
                raise ValueError(f"{r_idx + 1}번째 행이 리스트 형식이 아닙니다.")
            if len(row) != target_size: # raw(행렬)가 target_size * target_size 정방행렬인지 보는 분기(그 중 열)
                raise ValueError(
                    f"[{r_idx + 1}번째 행] 열 수 불일치: 기대 {target_size}열, 실제 {len(row)}열"
                )
            parsed_row: List[float] = []
            for c_idx, val in enumerate(row):
                if isinstance(val, bool) or not isinstance(val, (int, float)): # val이 블리언 타입이거나, val이 int/float 타입이 아닌지 확인하는 분기(블리언 타입은 int로 쳐주기 때문)
                    raise ValueError(
                        f"숫자가 아닌 값 포함 ({r_idx}, {c_idx}): {val!r}"
                    )
                num = float(val)
                if not math.isfinite(num):
                    raise ValueError(
                        f"유한하지 않은 숫자 포함 ({r_idx}, {c_idx}): {val!r}"
                    )
                parsed_row.append(num)
            parsed.append(parsed_row)

        return cls(target_size, parsed)


# ---------------------------------------------------------------------------
# 라벨 정규화
# ---------------------------------------------------------------------------
def normalize_filter_label(raw_key: Any) -> str:
    """필터 키('cross', 'x')를 표준 라벨('Cross', 'X')로 정규화합니다."""
    if not isinstance(raw_key, str):
        raise ValueError(f"필터 키가 문자열이 아닙니다: {raw_key!r}")
    key = raw_key.strip().lower()
    if key not in FILTER_LABEL_MAP:
        raise ValueError(f"지원하지 않는 필터 키입니다: {raw_key!r}")
    return FILTER_LABEL_MAP[key]


def normalize_expected_label(raw_label: Any) -> str:
    """expected 값('+', 'x')을 표준 라벨('Cross', 'X')로 정규화합니다."""
    if not isinstance(raw_label, str):
        raise ValueError(f"expected 값이 문자열이 아닙니다: {raw_label!r}")
    key = raw_label.strip().lower()
    if key not in EXPECTED_LABEL_MAP:
        raise ValueError(f"지원하지 않는 expected 라벨입니다: {raw_label!r}")
    return EXPECTED_LABEL_MAP[key]


# ---------------------------------------------------------------------------
# 핵심 MAC 연산, 비교 정책, 성능 측정
# ---------------------------------------------------------------------------
def mac_2d(pattern: Matrix2D, filter: Matrix2D) -> float: # 패턴과 필터를 비교하는 함수(MAC 연산을 통해)
    """반복문 기반 2차원 MAC(Multiply-Accumulate) 연산."""
    if pattern.size != filter.size:
        raise ValueError(
            f"크기 불일치: 패턴 {pattern.size}x{pattern.size}, "
            f"필터 {filter.size}x{filter.size}"
        )
    n = pattern.size
    p_data = pattern.data
    f_data = filter.data
    total = 0.0
    for i in range(n):
        p_row = p_data[i] # 패턴 데이터(이중 리스트)의 i번째 행(리스트)을 p_row에
        f_row = f_data[i] # 필터 데이터(이중 리스트)의 i번째 행(리스트)을 f_row에
        for j in range(n):
            total += p_row[j] * f_row[j]
    return total # MAC 결과값을 반환(CNN 합성곱 신경망의 픽셀 데이터)


def mac_1d(pattern_flat: List[float], filter_flat: List[float]) -> float:
    """1차원 평탄화 배열 기반 MAC 연산 (보너스: 메모리 접근 최적화)."""
    if len(pattern_flat) != len(filter_flat):
        raise ValueError(
            f"1D 배열 길이 불일치: 패턴 {len(pattern_flat)}, 필터 {len(filter_flat)}"
        )
    total = 0.0
    for i in range(len(pattern_flat)):
        total += pattern_flat[i] * filter_flat[i]
    return total


def compare_scores( # 점수 비교 함수
    score_a: float,
    score_b: float,
    label_a: str,
    label_b: str,
    epsilon: float = EPSILON,
) -> str:
    # 허용오차(epsilon) 기반 점수 비교 및 판정 분기 => 이유 : 파이썬에선 0.1 + 0.2 == 0.3이 False임(실제 계산값은 0.30000000000000004) 이를 막기 위해 1e-9 이하의 작은 차이는 무시하고 동점처리 하는 것
    # abs 함수는 절댓값(absolute)을 반환함 즉 score_a - score_b의 값이 양수던 음수던 무조건 양수가 되고 해당 양수가 1e-9(10억분의 1)보다 더 작은 값이라면 동점으로 치부하고 UNDECIDED 즉 판정 불가를 내는것
    if abs(score_a - score_b) < epsilon: # epsilon(앱실론, 여기선 아주 작은 허용 오차 기준값으로 사용) == 1e-9(10억분의 1)
        return UNDECIDED
    return label_a if score_a > score_b else label_b # score_a가 크면 label_a를 score_b가 크면 label_b를 반환


def measure_mac_2d( # MAC 연산 평균 시간 측정
    pattern: Matrix2D, flt: Matrix2D, repeats: int = DEFAULT_REPEATS # DEFAULT_REPEATS는 기본 반복 횟수로 10이라는 값을 가지고 있음
) -> float:
    """I/O를 제외한 순수 2D MAC 연산 평균 시간(ms)을 측정합니다."""
    start = time.perf_counter() # 측정 시작 시점(time.perf_counter()는 정밀도가 매우 높음)
    for _ in range(repeats): # _는 반복문 내부에서 i와 같은 변수를 사용하지 않고 그냥 반복만할때 사용
        mac_2d(pattern, flt)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 # time.perf_counter() - start의 결과는 초(s)이기 때문에 빠른 MAC연산에서는 보통 1초까지 안감 따라서 0.00000N/s와 같이 나오는데 결과를 보기 불편함으로 1000.0을 곱해 밀리초(ms)인 0.00N/ms와 같은 형태로 결과를 내기위해 1000.0을 곱하는 것
    return elapsed_ms / repeats # 반복없이 1번만 실행하고 반환하면 해당 실행시간에 OS가 다른 프로그램을 처리할 수 있어 실행시간에 편차가 심하고, 또 1회 연산은 너무 빨라 N 마이크로초안에 끝나는데 그러면 타이머 자체의 미세한 오차 비중이 커짐 따라서 10(또는 다른 값)번 반복하고 10으로 나누는 것


def measure_mac_1d(
    pattern_flat: List[float],
    filter_flat: List[float],
    repeats: int = DEFAULT_REPEATS,
) -> float:
    """I/O를 제외한 순수 1D MAC 연산 평균 시간(ms)을 측정합니다."""
    start = time.perf_counter()
    for _ in range(repeats):
        mac_1d(pattern_flat, filter_flat)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / repeats


def format_score(score: float, is_tie: bool) -> str: # 스코어 반환 함수
    # 동점(부동소수점 근접)일 때는 정밀 자릿수를, 일반 경우는 간결한 값을 출력하는 함수
    if is_tie: # is_tie가 True면 동점(판정불가)임 
        return f"{score:.16f}" # 소수점이하 16자리까지 표시
    return f"{round(score, 6)}" # 소수점이하 6자리에서 반올림


# ---------------------------------------------------------------------------
# 패턴 생성기 (보너스 과제 및 3x3 성능 벤치마크 재활용)
# ---------------------------------------------------------------------------
def generate_cross_matrix(n: int) -> Matrix2D:
    """N x N 십자가(Cross) 패턴을 생성합니다."""
    mat = Matrix2D(n)
    mid = n // 2
    for i in range(n):
        mat.set(mid, i, 1.0)
        mat.set(i, mid, 1.0)
    return mat


def generate_x_matrix(n: int) -> Matrix2D:
    """N x N X 패턴을 생성합니다."""
    mat = Matrix2D(n)
    for i in range(n):
        mat.set(i, i, 1.0)
        mat.set(i, n - 1 - i, 1.0)
    return mat


# ---------------------------------------------------------------------------
# 출력 유틸리티
# ---------------------------------------------------------------------------
def print_section(title: str) -> None: # title 매개변수 자료형 = str, 반환값 X
    print()
    print("-" * 44)
    print(f" {title}")
    print("-" * 44)


def print_perf_table(rows: List[Tuple[int, float, int]]) -> None:
    print(f"{'크기':<10}{'평균 시간(ms)':>14}{'연산 횟수':>12}")
    print("-" * 38)
    for size, avg_ms, ops in rows:
        size_str = f"{size}x{size}"
        print(f"{size_str:<10}{avg_ms:>14.6f}{ops:>12}")


# ---------------------------------------------------------------------------
# 모드 1: 사용자 입력 (3x3)
# ---------------------------------------------------------------------------
def input_matrix(name: str, size: int = 3) -> Matrix2D: # 반환값 = Matrix2D 인스턴스, 사용자 입력(패턴 or 필터)을 1행씩 검증(예외, 스키마(형식), 사이즈 등이 올바른지)
    print(f"\n{name} ({size}줄 입력, 공백 구분)")
    mat = Matrix2D(size)
    row_idx = 0
    while row_idx < size:
        raw_line = input(f"  {row_idx + 1}행 > ").strip() # 양쪽 끝 공백(또는 지정 문자)을 잘라내는 내장 함수
        tokens = raw_line.split() # 문자열을 리스트로(join의 반대)
        if len(tokens) != size:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
            )
            continue
        parsed_values: List[float] = []
        valid = True
        for token in tokens:
            try:
                val = float(token)
                if not math.isfinite(val): # 무한대나 NaN이면 안되니까 체크하고..
                    raise ValueError
                parsed_values.append(val)
            except ValueError:
                valid = False
                break
        if not valid:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
            )
            continue
        for col_idx, val in enumerate(parsed_values):
            mat.set(row_idx, col_idx, val) # 모든 검증을 거친 후 문제없는 행이면 set을 통해 mat의 멤버 변수(이중 리스트, 현재는 data를 생성자 파라미터로 넘겨주지 않았으므로 이중 리스트 내부 모든 값이 0.0임)인 self.data의 (row_idx, col_idx)좌표에 값(val) 삽입
        row_idx += 1
    return mat # 인스턴스 반환


def run_user_input_mode() -> None:
    # 모드 1 실행 흐름: 필터 A/B 입력 -> 저장 확인 -> 패턴 입력 -> MAC/판정 -> 성능 분석.
    print_section("[1] 필터 입력 (3x3)")
    filter_a = input_matrix("필터 A", 3) # self.data = 3*3 정방행렬인 인스턴스 생성
    filter_b = input_matrix("필터 B", 3) # self.data = 3*3 정방행렬인 인스턴스 생성

    print_section("[저장 확인]")
    print("✓ 필터 A 저장 완료:")
    print(filter_a.format_grid())
    print("✓ 필터 B 저장 완료:")
    print(filter_b.format_grid())

    print_section("[2] 패턴 입력 (3x3)")
    pattern = input_matrix("패턴", 3)

    score_a = mac_2d(pattern, filter_a) # pattern과 filter_a의 MAC 결과값을 score_a에
    score_b = mac_2d(pattern, filter_b) # pattern과 filter_b의 MAC 결과값을 score_b에
    decision = compare_scores(score_a, score_b, "A", "B", EPSILON) # A, B가 라벨임, 더 큰 스코어의 라벨을 반환 받음
    is_tie = decision == UNDECIDED # 만약 decision가 UNDECIDED(동점이여서 판정불가)면 is_tie에 True를 아니라면 False를

    avg_ms_a = measure_mac_2d(pattern, filter_a, DEFAULT_REPEATS) # pattern과 filter_a의 MAC 연산 평균 시간
    avg_ms_b = measure_mac_2d(pattern, filter_b, DEFAULT_REPEATS) # pattern과 filter_b의 MAC 연산 평균 시간
    avg_ms = (avg_ms_a + avg_ms_b) / 2.0 # 두 MAC 연산 평균 시간의 평균

    print_section("[3] MAC 결과")
    print(f"A 점수: {format_score(score_a, is_tie)}")
    print(f"B 점수: {format_score(score_b, is_tie)}")
    print(f"연산 시간(평균/{DEFAULT_REPEATS}회): {avg_ms:.6f} ms") # 이건 avg_ms_a, avg_ms_b의 각 연산 시간이 아닌 두 연산 시간의 평균임 
    if is_tie:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON:g})")
    else:
        print(f"판정: {decision}")

    print_section(f"[4] 성능 분석 (평균/{DEFAULT_REPEATS}회)")
    print_perf_table([(3, avg_ms, 3 * 3)])


# ---------------------------------------------------------------------------
# 모드 2: data.json 분석
# ---------------------------------------------------------------------------
def resolve_json_path(default_filename: str = "data.json") -> Optional[str]:
    """스크립트 디렉토리 또는 현재 작업 디렉토리에서 data.json 경로를 탐색합니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, default_filename),
        os.path.join(os.getcwd(), default_filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def parse_size_from_filter_key(filter_key: str) -> Optional[int]:
    match = re.match(r"^size_(\d+)$", filter_key)
    if not match:
        return None
    size = int(match.group(1))
    return size if size > 0 else None


def parse_size_from_pattern_key(pattern_key: str) -> Optional[int]:
    match = re.match(r"^size_(\d+)_.+$", pattern_key)
    if not match:
        return None
    size = int(match.group(1))
    return size if size > 0 else None


def load_filters_from_json(
    raw_filters: Any,
) -> Tuple[Dict[int, Dict[str, Matrix2D]], Dict[int, str]]:
    """JSON의 filters 항목을 정규화/검증하여 로드합니다."""
    loaded: Dict[int, Dict[str, Matrix2D]] = {}
    errors: Dict[int, str] = {}

    if not isinstance(raw_filters, dict):
        print("✗ filters 섹션이 올바른 JSON 객체 형식이 아닙니다.")
        return loaded, errors

    for key, filter_group in raw_filters.items():
        size = parse_size_from_filter_key(key)
        if size is None:
            print(f"✗ 잘못된 필터 키 형식 무시: {key!r}")
            continue
        if not isinstance(filter_group, dict):
            msg = f"{key} 필터 항목이 객체 형식이 아닙니다."
            errors[size] = msg
            print(f"✗ {key} 로드 실패: {msg}")
            continue

        normalized_pair: Dict[str, Matrix2D] = {}
        try:
            for raw_label, raw_matrix in filter_group.items():
                std_label = normalize_filter_label(raw_label)
                normalized_pair[std_label] = Matrix2D.from_list(
                    raw_matrix, expected_size=size
                )
            missing = [lbl for lbl in ("Cross", "X") if lbl not in normalized_pair]
            if missing:
                raise ValueError(f"필수 필터 누락: {', '.join(missing)}")
            loaded[size] = normalized_pair
            print(f"✓ {key:<7} 필터 로드 완료 (Cross, X)")
        except ValueError as exc:
            errors[size] = str(exc)
            print(f"✗ {key:<7} 필터 로드 실패: {exc}")

    return loaded, errors


def run_json_analysis_mode(json_path: Optional[str] = None) -> None:
    """모드 2 실행 흐름: 필터 로드 -> 패턴 분석 -> 성능 분석 -> 결과 요약."""
    target_path = json_path or resolve_json_path("data.json")
    if target_path is None or not os.path.isfile(target_path):
        print("\n오류: data.json 파일을 찾을 수 없습니다. 실행 경로를 확인하세요.")
        return

    try:
        with open(target_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"\n오류: data.json 읽기/파싱 실패 - {exc}")
        return

    if not isinstance(payload, dict):
        print("\n오류: data.json 최상위 구조가 JSON 객체가 아닙니다.")
        return

    print_section("[1] 필터 로드")
    filters, filter_errors = load_filters_from_json(payload.get("filters"))

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    raw_patterns = payload.get("patterns")
    if not isinstance(raw_patterns, dict) or len(raw_patterns) == 0:
        print("분석할 patterns 항목이 없거나 형식이 올바르지 않습니다.")
        return

    total_count = 0
    pass_count = 0
    failed_cases: List[Tuple[str, str]] = []
    sample_pattern_by_size: Dict[int, Matrix2D] = {}

    for case_id, case_body in raw_patterns.items():
        total_count += 1
        print(f"-- {case_id} ---")

        size = parse_size_from_pattern_key(case_id)
        if size is None:
            reason = "스키마 오류: 패턴 키에서 크기 N 추출 실패 (size_{N}_{idx} 형식 필요)"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        if not isinstance(case_body, dict):
            reason = "스키마 오류: 패턴 항목이 객체 형식이 아님"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        if size not in filters:
            detail = filter_errors.get(size, f"size_{size} 필터 미존재")
            reason = f"필터 오류: {detail}"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        try:
            expected_label = normalize_expected_label(case_body.get("expected"))
        except ValueError as exc:
            reason = f"라벨 오류: {exc}"
            print(f"판정: ERROR | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        try:
            pattern = Matrix2D.from_list(
                case_body.get("input"), expected_size=size
            )
        except ValueError as exc:
            reason = f"크기/스키마 불일치 ({size}x{size} 기대): {exc}"
            print(f"판정: ERROR | expected: {expected_label} | FAIL ({reason})")
            failed_cases.append((case_id, reason))
            continue

        if size not in sample_pattern_by_size:
            sample_pattern_by_size[size] = pattern

        cross_filter = filters[size]["Cross"]
        x_filter = filters[size]["X"]

        cross_score = mac_2d(pattern, cross_filter)
        x_score = mac_2d(pattern, x_filter)
        decision = compare_scores(cross_score, x_score, "Cross", "X", EPSILON)
        is_tie = decision == UNDECIDED

        print(f"Cross 점수: {format_score(cross_score, is_tie)}")
        print(f"X 점수: {format_score(x_score, is_tie)}")

        if decision == expected_label:
            pass_count += 1
            print(f"판정: {decision} | expected: {expected_label} | PASS")
        else:
            if is_tie:
                reason = f"동점({UNDECIDED}) 처리 규칙에 따라 FAIL (|Cross-X| < {EPSILON:g})"
                print(
                    f"판정: {decision} | expected: {expected_label} | FAIL (동점 규칙)"
                )
            else:
                reason = f"예측 불일치 (판정: {decision}, expected: {expected_label})"
                print(f"판정: {decision} | expected: {expected_label} | FAIL")
            failed_cases.append((case_id, reason))

    # [3] 성능 분석 (3x3 포함, data.json의 5x5 / 13x13 / 25x25)
    print_section(f"[3] 성능 분석 (평균/{DEFAULT_REPEATS}회)")
    perf_rows: List[Tuple[int, float, int]] = []

    # 3x3 기본 벤치마크 (패턴 생성기 활용)
    pat_3 = generate_cross_matrix(3)
    flt_3 = generate_cross_matrix(3)
    avg_3 = measure_mac_2d(pat_3, flt_3, DEFAULT_REPEATS)
    perf_rows.append((3, avg_3, 3 * 3))

    for size in sorted(filters.keys()):
        if size == 3:
            continue
        pat = sample_pattern_by_size.get(size, generate_cross_matrix(size))
        flt = filters[size]["Cross"]
        avg_ms = measure_mac_2d(pat, flt, DEFAULT_REPEATS)
        perf_rows.append((size, avg_ms, size * size))

    print_perf_table(perf_rows)

    # [4] 결과 요약
    fail_count = len(failed_cases)
    print_section("[4] 결과 요약")
    print(f"총 테스트: {total_count}개")
    print(f"통과: {pass_count}개")
    print(f"실패: {fail_count}개")

    if failed_cases:
        print("\n실패 케이스:")
        for case_id, reason in failed_cases:
            print(f"- {case_id}: {reason}")


# ---------------------------------------------------------------------------
# 보너스 과제: 1D 최적화 성능 비교 & N x N 패턴 생성기
# ---------------------------------------------------------------------------
def run_optimization_benchmark() -> None:
    """2D MAC vs 1D 평탄화 MAC 동일 조건 성능 비교."""
    print_section("[보너스 1] 2D vs 1D 메모리 접근 최적화 비교")
    repeats = 100
    sizes = [3, 5, 13, 25]

    print(f"(각 크기별 동일 입력, {repeats}회 반복 평균 측정)\n")
    print(
        f"{'크기':<8}{'2D 평균(ms)':>14}{'1D 평균(ms)':>14}"
        f"{'개선율(%)':>12}{'연산 횟수':>10}"
    )
    print("-" * 60)

    for n in sizes:
        pat = generate_cross_matrix(n)
        flt = generate_cross_matrix(n)
        pat_flat = pat.to_flat()
        flt_flat = flt.to_flat()

        # 결과 동일성 검증
        assert abs(mac_2d(pat, flt) - mac_1d(pat_flat, flt_flat)) < EPSILON

        ms_2d = measure_mac_2d(pat, flt, repeats)
        ms_1d = measure_mac_1d(pat_flat, flt_flat, repeats)
        improvement = ((ms_2d - ms_1d) / ms_2d * 100.0) if ms_2d > 0 else 0.0

        size_str = f"{n}x{n}"
        print(
            f"{size_str:<8}{ms_2d:>14.6f}{ms_1d:>14.6f}"
            f"{improvement:>11.2f}%{n * n:>10}"
        )


def run_pattern_generator_mode() -> None:
    """크기 N 입력 시 Cross/X 패턴을 자동 생성하고 MAC 판정/성능 측정에 재활용."""
    print_section("[보너스 2] N x N 패턴 생성기")
    while True:
        raw = input("생성할 패턴 크기 N 입력 (홀수 권장, 3~25): ").strip()
        try:
            n = int(raw)
            if n < 3 or n > 50:
                print("입력 오류: 3 이상 50 이하의 정수를 입력하세요.")
                continue
            break
        except ValueError:
            print("입력 오류: 정수를 입력하세요.")

    cross_mat = generate_cross_matrix(n)
    x_mat = generate_x_matrix(n)

    print(f"\n[생성된 {n}x{n} Cross 패턴]")
    print(cross_mat.format_grid())
    print(f"\n[생성된 {n}x{n} X 패턴]")
    print(x_mat.format_grid())

    # 생성된 패턴으로 즉시 MAC 및 성능 검증
    cross_on_cross = mac_2d(cross_mat, cross_mat)
    cross_on_x = mac_2d(cross_mat, x_mat)
    decision = compare_scores(cross_on_cross, cross_on_x, "Cross", "X", EPSILON)
    avg_ms = measure_mac_2d(cross_mat, cross_mat, DEFAULT_REPEATS)

    print(f"\n[생성 패턴 MAC 검증: Cross 입력 기준]")
    print(f"Cross 필터 점수: {cross_on_cross:g}")
    print(f"X 필터 점수: {cross_on_x:g}")
    print(f"판정: {decision}")
    print(f"평균 연산 시간({DEFAULT_REPEATS}회): {avg_ms:.6f} ms (연산 횟수: {n * n})")


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------
def main() -> None: # 반환값 None
    print("=== Mini NPU Simulator ===")
    while True:
        # 메뉴 출력
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. [보너스] 2D vs 1D 최적화 성능 비교")
        print("4. [보너스] N x N 패턴 생성기")
        print("0. 종료")

        try:
            choice = input("선택: ").strip()
        except (EOFError, KeyboardInterrupt): # Ctrl + c, Ctrl + z가 들어오면 종료
            print("\n프로그램을 종료합니다.")
            break

        # 사용자 선택별 분기 선택
        if choice == "1":
            run_user_input_mode()
        elif choice == "2":
            run_json_analysis_mode()
        elif choice == "3":
            run_optimization_benchmark()
        elif choice == "4":
            run_pattern_generator_mode()
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 0~4 사이의 숫자를 입력하세요.")


if __name__ == "__main__":
    main() #main 함수로 실행