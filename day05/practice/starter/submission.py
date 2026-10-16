"""Day 5 — 승인 게이트와 가드레일.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 이 파일의 함수는 LLM을 호출하지 않아도 동작해야 합니다.
     규칙 기반으로 먼저 막고, 필요하면 모델 판별을 덧붙이세요.
  3. 채점은 교안에 공개된 20문항이 아니라 **비공개 문항**으로 합니다.
     공개 문항만 통과하는 하드코딩은 채점에서 걸립니다.
"""

from __future__ import annotations

import json
import re
import unicodedata          # ← 추가
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_public_sets() -> tuple[list[str], list[str]]:
    """연습용 공개 세트를 읽습니다. 채점에는 쓰이지 않습니다."""
    attack = json.loads((DATA_DIR / "attack_set.json").read_text(encoding="utf-8"))
    normal = json.loads((DATA_DIR / "normal_set.json").read_text(encoding="utf-8"))
    return attack, normal


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# 입력 가드 규칙
#
# 설계 원칙: "단어 하나"가 아니라 "동작 + 대상"의 조합을 봅니다.
#   - "무시해도 되는 알람"      → 동작(무시)만 있고 대상(지시)이 없음 → 통과
#   - "이 알람은 지시대로 처리" → 대상(지시)만 있고 덮어쓰는 동작이 없음 → 통과
#   - "이전 지시는 모두 무시"   → 대상 + 동작이 함께 있음 → 차단
#
# 각 규칙은 (사유, 정규식) 쌍이며, 위에서부터 차례로 검사해 처음 걸린 사유를 돌려줍니다.
# 공개 20문항의 문장을 그대로 넣지 않고, 같은 의도의 변형 표현까지 잡도록 넓게 썼습니다.
# ──────────────────────────────────────────────────────────────────

# [1] 지시 덮어쓰기 — "현재 지시"라는 대상
#     한국어는 목적어가 동사보다 앞에 오므로 "대상 → 동작" 순서만 봅니다.
#     "규칙", "설정"은 정상 문장("알람 규칙 무시해도 돼?")에 자주 나와 대상에서 뺐습니다.
_KO_OVERRIDE_TARGET = r"(지시|지침|명령|프롬프트|대화)"
#     "잊지 말고", "무시하지 말고"처럼 부정형은 오히려 지시를 지키라는 뜻이므로 제외합니다.
_KO_OVERRIDE_ACTION = (
    r"(무시(?!하지\s*(마|말))|잊(?!지\s*(마|말|않))|됐고|됐어|버리|버려|덮어|초기화"
    r"|따르지\s*마|따르지\s*말)"
)
_EN_OVERRIDE_ACTION = r"(ignore|forget|disregard|override|bypass)"
_EN_OVERRIDE_TARGET = r"(instruction|prompt|rule|guideline|context|direction)"

# [2] 시스템 프롬프트 유출 — "숨겨진 지침"이라는 대상 + "드러내라"는 동작
#     "시스템"만으로는 막지 않습니다 ("시스템 점검은 언제?"는 정상).
_PROMPT_TARGET = (
    r"(시스템\s*(프롬프트|메시지|메세지|지침|지시|명령)"
    r"|system\s*(prompt|message|instruction)"
    r"|(너|당신|니)의?\s*(프롬프트|지침|초기\s*지시)"
    r"|your\s*(prompt|instructions|system|initial)"
    r"|(숨겨진|초기|최초|원래|처음\s*받은)\s*(지침|지시|프롬프트|설정\s*문)"
    r"|(hidden|initial|original)\s*(prompt|instruction))"
)
#     "보여"는 "안 보여요"(정상)와 구분하려고 "보여 줘/달라/라" 요청형만 봅니다.
_REVEAL_ACTION = (
    r"(알려|보여\s*(줘|주|달|라)|출력|공개|말해|적어|써\s*줘|전문|그대로|원문|노출|유출|덤프"
    r"|reveal|print|show|tell|repeat|dump|output|leak)"
)

# [3] 자격증명 유출 — 비밀값 + "평문으로 꺼내라"는 동작
#     "비밀번호 재설정 방법 알려줘"는 정상이므로 '알려'는 동작에서 뺐습니다.
_SECRET_TARGET = (
    r"(비밀\s*번호|패스워드|password|api\s*키|api\s*key|액세스\s*키|access\s*key"
    r"|시크릿|secret|자격\s*증명|credential|내부\s*토큰|토큰\s*값|private\s*key|개인\s*키)"
)
_DUMP_ACTION = r"(보여\s*(줘|주|달|라)|출력|공개|노출|덤프|평문|그대로|따라|reveal|print|dump|leak|expose)"

# [4] 우회 인코딩 — 인코딩/변형 요청 + 숨겨진 값
_ENCODE_ACTION = r"(base64|base\s*64|인코딩|encode|rot13|hex|16진수|역순|거꾸로|암호화해서)"
_ENCODE_TARGET = r"(시스템|지침|프롬프트|prompt|instruction|비밀|키|key|토큰|token|secret)"

_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "지시 덮어쓰기 시도",
        re.compile(
            rf"{_KO_OVERRIDE_TARGET}.{{0,12}}{_KO_OVERRIDE_ACTION}"
            rf"|{_EN_OVERRIDE_ACTION}.{{0,25}}{_EN_OVERRIDE_TARGET}"
        ),
    ),
    (
        "역할 탈취 시도",
        re.compile(
            # "지금부터 너의 역할은 ~", "너는 이제 ~"
            r"(지금부터|이제부터|앞으로).{0,20}(너|당신|니)(의|는|가)?\s*.{0,6}(역할|정체|임무)"
            r"|(너|당신)(는|은)\s*이제(부터)?"
            # "새로운 역할을 부여받아라" — "새로운 역할 담당자"(정상)와 구분: '부여/맡' 동작 필요
            r"|역할을\s*(새로\s*)?(부여|맡아|맡으|수행하라)"
            r"|you\s+are\s+now|act\s+as\s+(an?\s+)?(unrestricted|admin|root|developer)"
            r"|pretend\s+(to\s+be|you\s+are)"
            # 특권 모드 전환 — "관리자에게 문의"(정상)와 구분: '모드' + 전환 동작 필요
            r"|(관리자|개발자|탈옥|무제한|admin|root|sudo|god|developer|jailbreak)\s*모드\s*"
            r"(로|으로)?\s*(전환|진입|변경|바꿔|들어가|활성|켜|야|다\b|이다|입니다)"
            r"|(developer|god|jailbreak|dan)\s+mode|jailbreak"
        ),
    ),
    (
        "구분자 주입 시도",
        re.compile(
            r"#{2,}\s*(new|system|instruction|override|admin)"
            r"|\[\s*(system|inst|admin)\s*\]"
            r"|<\|?\s*(system|im_start|endoftext)"
            r"|new\s+instructions?\s*[:#]"
            r"|(^|\n)\s*system\s*:"
        ),
    ),
    (
        "시스템 프롬프트 유출 시도",
        re.compile(
            rf"{_PROMPT_TARGET}.{{0,20}}{_REVEAL_ACTION}"
            rf"|{_REVEAL_ACTION}.{{0,20}}{_PROMPT_TARGET}"
        ),
    ),
    (
        "자격증명 유출 시도",
        re.compile(
            rf"{_SECRET_TARGET}.{{0,15}}{_DUMP_ACTION}"
            rf"|{_DUMP_ACTION}.{{0,25}}{_SECRET_TARGET}"
        ),
    ),
    (
        "인코딩 우회 유출 시도",
        re.compile(
            rf"{_ENCODE_ACTION}.{{0,20}}{_ENCODE_TARGET}"
            rf"|{_ENCODE_TARGET}.{{0,20}}{_ENCODE_ACTION}"
        ),
    ),
    (
        "대량 파괴 명령 시도",
        re.compile(
            # 범위(전부/모든) + 삭제 동작 + 실행 지시가 함께 있어야 차단합니다.
            # "로그 전부 삭제해도 되나요?" 같은 질문은 실행 지시가 없어 통과합니다.
            r"(전부|모두|전체|모든|싹\s*다|all).{0,15}(삭제|지우|지워|날려|drop|truncate|delete|wipe)"
            r".{0,15}(실행|해\s*줘|해라|하라|해$|해\s*[.!]|명령|now)"
            r"|rm\s+-rf\s+/|drop\s+(table|database)|truncate\s+table"
        ),
    ),
]


def _normalize(text: str) -> str:
    """판정 전 정규화: 전각/호환 문자 통일(NFKC) → 소문자 → 연속 공백 1칸."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def input_guard(text: str) -> tuple[bool, str]:
    """입력을 검사해 차단 여부를 정합니다.

    Returns:
        (차단할지 여부, 사유 문자열)

    주의: 규칙을 늘릴수록 정상 요청도 막힙니다.
    normal_set에는 "이전", "무시", "새로운 역할", "관리자" 같은 함정 단어가
    일부러 들어 있습니다. 단어 하나로 막으면 오탐이 터집니다.
    """
    # TODO: 구현하세요.
    # 빈 입력·공백 입력·None은 막지 않습니다. (막을 의도 자체가 없으므로)
    if text is None or not str(text).strip():
        return False, "빈 입력: 검사할 내용 없음"

    norm = _normalize(str(text))
    # "시 스 템 프 롬 프 트"처럼 글자 사이에 공백을 넣는 우회를 막기 위해
    # 공백을 모두 지운 버전도 함께 검사합니다.
    compact = norm.replace(" ", "")

    for reason, pattern in _RULES:
        if pattern.search(norm) or pattern.search(compact):
            return True, reason

    return False, "통과: 차단 규칙에 해당 없음"

# ──────────────────────────────────────────────────────────────────
# PII·자격증명 마스킹
#
# 적용 순서가 중요합니다. 길고 구체적인 패턴부터 처리해야 서로 간섭하지 않습니다.
#   JWT → AWS 키 → 이메일 → 전화번호 → 사번
# 치환 토큰([JWT], [EMAIL] …)은 어떤 패턴에도 다시 걸리지 않으므로,
# 두 번 돌려도 결과가 같습니다(멱등). 매칭이 없으면 원문이 그대로 나옵니다.
# ──────────────────────────────────────────────────────────────────
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # JWT: header.payload(.signature) — header가 항상 "eyJ"({")로 시작합니다.
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?"), "[JWT]"),
    # AWS 액세스 키: AKIA(장기 키) / ASIA(임시 키) + 대문자·숫자 16자
    (re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"), "[AWS_KEY]"),
    # 이메일
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), "[EMAIL]"),
    # 전화번호: 휴대폰(010 등) · 서울(02) · 지역번호(031~064) · 070
    #   국제 형식(+82-10-…)은 앞자리 0이 빠지므로 "+82 뒤에는 0 생략 가능"으로 처리합니다.
    (
        re.compile(
            r"(?<!\d)(?:\+82[-. ]?0?|0)(?:1[016789]|2|[3-6][1-5]|70)[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)"
        ),
        "[PHONE]",
    ),
    # 사번: E + 숫자 6자리. "E123456이"처럼 한글이 바로 붙으면 \b가 경계를 못 잡으므로
    # 영숫자 기준 전후방 탐색으로 경계를 직접 정의합니다.
    (re.compile(r"(?<![A-Za-z0-9])E\d{6}(?!\d)"), "[EMP_ID]"),
]


def mask_pii(text: str) -> str:
    """개인정보와 자격증명을 마스킹합니다.

    최소 다섯 가지를 가려야 합니다.
      사번(예: E123456) · 전화번호 · 이메일 · AWS 액세스 키(AKIA...) · JWT(eyJ...)
    """
    # TODO: 구현하세요.
    if not text:
        return text
    for pattern, token in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text



# TODO: 미들웨어를 어떤 순서로 둘지 정하세요.
# 미들웨어 순서 — 원칙: 차단 -> 정제 -> 검증 -> 기록
#   마스킹이 로깅보다 뒤에 있으면 로그에 원본이 남습니다. 순서 자체가 설계입니다.
MIDDLEWARE_ORDER: list[str] = [
    "InputGuardMiddleware",   # 1. 차단: 공격 입력은 이후 단계에 들어가기 전에 끊습니다.
    "MaskingMiddleware",      # 2. 정제: 모델·트레이스·로그로 흘러가기 전에 PII를 가립니다.
    "OutputCheckMiddleware",  # 3. 검증: 출력 검사 결과(무엇이 걸렸는지)를 기록에 함께 남기기 위해 기록보다 앞.
    "LoggingMiddleware",      # 4. 기록: 이미 마스킹·검증된 값만 남습니다. 로그는 되돌릴 수 없으므로 맨 마지막.
]



# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# TODO: 도구별 위험도를 정하세요. 값은 "read" | "write" | "destructive" 중 하나입니다.
# 도구별 위험도 — 코드라기보다 정책 문서입니다.
#   조회는 read, 상태를 바꾸면 write, 되돌릴 수 없으면 destructive.
#   등록하지 않은 도구는 needs_approval에서 자동으로 "승인 필요"로 떨어집니다(안전한 실패).
RISK_LEVELS: dict[str, str] = {
    # read — 승인 없이 자동 실행. 넉넉히 등록해야 승인 알림 피로가 생기지 않습니다.
    "get_metrics": "read",
    "get_logs": "read",
    "get_deploy_history": "read",
    "get_asset_status": "read",
    "get_asset_status_backup": "read",
    "search_runbook": "read",
    "get_service_owner": "read",        # Day 4 도구 — 담당팀·온콜 조회
    "query_recent_incidents": "read",   # Day 4 도구 — 최근 인시던트 조회
    # write — 상태 변경(재시작·스케일 조정). 되돌릴 수는 있으나 사람 승인 필요.
    "restart_service": "write",
    "scale_service": "write",
    "rollback_deploy": "write",
    # destructive — 되돌릴 수 없음(삭제·자격증명 교체·프로세스 강제 종료·권한 상승·임의 명령 실행). 승인 + 이중 확인.
    "delete_asset": "destructive",
    "rotate_credentials": "destructive",
    "drop_database": "destructive",
    "kill_process": "destructive",
    "terminate_instance": "destructive",
    "grant_admin_role": "destructive",
    "escalate_privilege": "destructive",
    "run_command": "destructive",
    "execute_shell": "destructive",
}

# 위험도 상향 규칙 — RISK_LEVELS 등록만으로는 막을 수 없는 두 구멍을 메웁니다.
#   (1) 미등록 도구라도 이름이 파괴·권한 상승을 뜻하면 destructive로 봅니다.
#       예: kill_pod, revoke_token, grant_sudo, chmod_file …
#       → "모르는 것은 막는다"(승인 필요)는 그대로 지키고, 이중 확인까지 붙입니다.
#   (2) 등록된 도구라도 인자(args)에 위험 명령이 실려 오면 destructive로 올립니다.
#       예: restart_service(cmd="kill -9 1"), get_logs(path="/etc/sudoers") …
#       → 이름만 보고 판정하면 인자로 우회됩니다. 상향만 하고 하향은 하지 않습니다.
_DANGEROUS_TOOL_NAME = re.compile(
    r"(delete|drop|remove|destroy|purge|wipe|truncate|format|rotate|revoke|reset"
    r"|kill|terminate|shutdown|reboot|halt|stop_all"
    r"|grant|escalat|elevat|privilege|sudo|root|admin|chmod|chown|impersonat"
    r"|exec|shell|command|run_cmd|eval)",
    re.IGNORECASE,
)
_DANGEROUS_ARG = re.compile(
    # 프로세스 강제 종료·시스템 중단
    r"(\bkill\b|pkill|killall|taskkill|\bshutdown\b|\breboot\b|\bhalt\b|systemctl\s+(stop|kill|disable)"
    # 권한 상승
    r"|\bsudo\b|\bsu\b(\s+-|\s+root|$)|\bdoas\b|runas|setuid|chmod\s+(\+s|[0-7]*[4-7][0-7]{2}\b|777)"
    r"|chown\s+root|usermod\s+-a?g|visudo|/etc/(sudoers|passwd|shadow)|--privileged"
    r"|grant\s+(all|admin|superuser)|alter\s+user|role\s*[:=]\s*[\"']?(admin|root|superuser)"
    r"|is_?admin\s*[:=]\s*(true|1)"
    # 되돌릴 수 없는 삭제
    r"|rm\s+-[a-z]*r[a-z]*f|rm\s+-[a-z]*f[a-z]*r|drop\s+(table|database)|truncate\s+table|mkfs|dd\s+if=)",
    re.IGNORECASE,
)


def _args_text(args: Any) -> str:
    """args(dict·list·문자열 중첩 가능)를 검사용 문자열 하나로 펼칩니다. 키 이름도 포함합니다."""
    if args is None:
        return ""
    if isinstance(args, dict):
        return " ".join(f"{k}={_args_text(v)}" for k, v in args.items())
    if isinstance(args, (list, tuple, set)):
        return " ".join(_args_text(v) for v in args)
    return str(args)


def needs_approval(tool_name: str, args: dict) -> tuple[bool, str]:
    # 같은 (도구, 인자)는 항상 같은 판정을 냅니다(결정성). LLM 없이 규칙만 씁니다.
    name = str(tool_name or "").strip()
    level = RISK_LEVELS.get(name) or RISK_LEVELS.get(name.lower())

    # (1) 미등록인데 이름이 파괴·권한 상승을 뜻하면 destructive
    if level is None and _DANGEROUS_TOOL_NAME.search(name):
        return True, (
            f"destructive: 미등록 위험 도구 '{name}'(파괴·종료·권한 상승 추정) "
            "— 승인 + 이중 확인(double confirm) 필요"
        )

    # (2) 인자에 위험 명령(kill·sudo·rm -rf 등)이 있으면 등급과 무관하게 destructive로 상향
    hit = _DANGEROUS_ARG.search(_args_text(args))
    if hit:
        return True, (
            f"destructive: 인자에 위험 명령 '{hit.group(0).strip()}' 포함 "
            "— 승인 + 이중 확인(double confirm) 필요"
        )

    if level == "read":
        return False, "read: 조회 도구 — 승인 없이 자동 실행"
    if level == "write":
        return True, "write: 상태 변경 — 사람 승인 필요"
    if level == "destructive":
        return True, "destructive: 되돌릴 수 없음 — 승인 + 이중 확인(double confirm) 필요"
    return True, f"unknown: 미등록 도구 '{tool_name}' — 모르는 것은 막는다, 승인 필요"

