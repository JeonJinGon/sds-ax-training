"""Day 4 — 운영 지원 MCP 서버와 장애 복구.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 도구는 예외를 밖으로 던지지 않습니다. 무엇이 잘못됐는지 설명하는 문자열을 돌려주세요.
     그 문자열은 모델이 읽고 다음 행동을 정하는 프롬프트입니다.
  3. 자격증명은 절대 반환값이나 로그에 남기지 마세요.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

DATA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "ops_assets.json").read_text(
        encoding="utf-8"
    )
)

# 교육용 더미 자격증명입니다. 이 값이 로그나 반환값에 남으면 채점에서 감점됩니다.
BACKEND_TOKEN = "sk-internal-dummy-1234"

MAX_RETRIES = 3


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

def search_runbook(symptom: str) -> str:
    """증상 키워드로 대응 런북을 찾습니다."""
    # TODO: DATA["runbooks"]에서 찾아 문자열로 반환하세요.
    #   없는 키워드면 예외가 아니라 "어떤 키워드가 있는지" 알려 주는 문자열을 반환합니다.
    runbooks = DATA.get("runbooks", {})
    available = ", ".join(sorted(runbooks))      # 안내용: 쓸 수 있는 키워드 목록
    text = str(symptom or "").strip().lower()
    if not text:
        return f"증상 키워드가 비어 있습니다. 쓸 수 있는 값: {available}"

    
    for key, steps in runbooks.items():
        if key.lower() in text or text in key.lower():
            return f"[런북: {key}] {steps}"

    
    return f"'{symptom}'은(는) 등록되지 않은 증상 키워드입니다. 쓸 수 있는 값: {available}"
    raise NotImplementedError("search_runbook 을 구현하세요.")


def get_service_owner(service: str) -> str:
    """서비스의 담당팀과 온콜 담당자를 조회합니다."""
    # TODO: DATA["owners"] 에서 조회하세요.
    owners = DATA.get("owners", {})
    available = ", ".join(sorted(owners))
    name = str(service or "").strip()
    if not name:
        return f"서비스 이름이 비어 있습니다. 조회 가능한 서비스: {available}"

    info = owners.get(name)
    if not info:
        return f"'{name}'은(는) 등록되지 않은 서비스입니다. 조회 가능한 서비스: {available}"

    return (
        f"[{name} 담당] 팀: {info.get('team', '-')}, "
        f"온콜: {info.get('oncall', '-')}, 슬랙: {info.get('slack', '-')}"
    )
    raise NotImplementedError("get_service_owner 를 구현하세요.")


def query_recent_incidents(service: str) -> str:
    """서비스의 최근 인시던트 이력을 조회합니다."""
    # TODO: DATA["incidents"] 에서 조회하세요.
    incidents = DATA.get("incidents", {})
    known = set(DATA.get("owners", {})) | set(incidents)
    available = ", ".join(sorted(known))
    name = str(service or "").strip()
    if not name:
        return f"서비스 이름이 비어 있습니다. 조회 가능한 서비스: {available}"
    if name not in known:
        return f"'{name}'은(는) 등록되지 않은 서비스입니다. 조회 가능한 서비스: {available}"

    items = incidents.get(name) or []
    if not items:
        return f"[{name}] 최근 인시던트 이력이 없습니다."

    lines = [f"[{name} 최근 인시던트 — 최신순]"]
    for it in items:
        lines.append(
            f"- {it.get('id')} ({it.get('at')}) {it.get('title')} / 원인: {it.get('cause')}"
        )
    return "\n".join(lines)
    raise NotImplementedError("query_recent_incidents 를 구현하세요.")


# TODO: MCP로 노출할 도구 명세를 채우세요.
#   각 항목: {"name": ..., "description": 30자 이상, "args": {인자명: 설명}}
#   description에는 '언제 쓰는지'와 '언제 쓰지 말아야 하는지'를 적으세요.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_runbook",
        "description": (
            "증상 키워드로 장애 대응 런북(조치 순서)을 찾습니다. 장애 증상은 알지만 "
            "무엇부터 확인할지 모를 때 씁니다. 담당자나 과거 이력을 찾을 때는 쓰지 마세요."
        ),
        "args": {"symptom": "증상 키워드. 쓸 수 있는 값: 5xx, oom, cert"},
    },
    {
        "name": "get_service_owner",
        "description": (
            "서비스의 담당팀, 온콜 담당자, 슬랙 채널을 조회합니다. 장애를 누구에게 "
            "에스컬레이션할지 정할 때 씁니다. 장애 원인이나 조치 방법을 찾을 때는 쓰지 마세요."
        ),
        "args": {"service": "서비스 이름. 예: payment-api, auth-service, gateway"},
    },
    {
        "name": "query_recent_incidents",
        "description": (
            "서비스의 최근 인시던트 이력(ID, 날짜, 제목, 원인)을 조회합니다. 비슷한 장애가 "
            "전에 있었는지 확인할 때 씁니다. 현재 상태나 담당자를 알고 싶을 때는 쓰지 마세요."
        ),
        "args": {"service": "서비스 이름. 예: payment-api, auth-service"},
    },
]

# TODO: 이름 -> 실제 함수 매핑. TOOL_SPECS의 name과 키가 정확히 일치해야 합니다.
TOOLS_IMPL: dict[str, Any] = {
    "search_runbook": search_runbook,
    "get_service_owner": get_service_owner,
    "query_recent_incidents": query_recent_incidents,
}


def _default_fetch(asset_id: str) -> dict:
    """기본 자산 조회 경로입니다. 실제로는 외부 시스템을 호출한다고 가정합니다."""
    asset = DATA["assets"].get(asset_id)
    if not asset:
        raise LookupError(f"asset not found: {asset_id}")
    return asset


def get_asset_status(asset_id: str, fetch: Callable[[str], dict] | None = None) -> str:
    """자산 상태를 조회합니다. 실패하면 재시도하고, 그래도 안 되면 대체 경로를 씁니다.

    반드시 지킬 것:
      - 어떤 경우에도 예외를 밖으로 던지지 않습니다.
      - 대체 경로로 넘어갔다면 반환 문자열에 'get_asset_status_backup' 이름을 남깁니다.
        모델이 다음에 무엇을 할지 알아야 하기 때문입니다.
      - BACKEND_TOKEN 값을 반환 문자열에 넣지 마세요.

    Args:
        asset_id: 조회할 자산 ID
        fetch: 외부 조회 함수. 채점기가 실패하는 함수를 넣습니다. None이면 기본 경로.
    """
    fetch = fetch or _default_fetch
    # TODO: 구현하세요.
    #   1) fetch를 호출한다
    #   2) 실패하면 재시도한다 (심화를 했다면 classify_failure로 전략을 나눈다)
    #   3) 끝내 실패하면 get_asset_status_backup 으로 넘어간다
    # TODO: 구현하세요.
    asset_id = str(asset_id or "").strip()
    if not asset_id:
        return "자산 ID가 비어 있습니다. SRV-001 같은 형식의 자산 ID를 넣어 주세요."

    attempts = 0
    last_kind = "unknown"   # 마지막 실패의 분류 (retryable / backoff / fatal)
    last_error = ""         # 마지막 예외의 '타입 이름'만 기록 (메시지는 남기지 않음)

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        try:
            asset = fetch(asset_id)
            # 응답 모양 검사: status가 없으면 README 로그의 '스키마 오류'와 같은 상황입니다.
            if not isinstance(asset, dict) or "status" not in asset:
                raise ValueError("schema validation failed: field 'status' missing")
            # 성공 → 되살아난 실제 값을 반환합니다. (폴백 문구로 덮어쓰지 않음)
            # 필요한 필드만 골라 담아, 응답에 섞여 있을지 모를 민감 정보가 새지 않게 합니다.
            return (
                f"[{asset_id} 상태 — 기본 경로, 시도 {attempts}회] "
                f"상태: {asset.get('status')}, 이름: {asset.get('name', '-')}, "
                f"담당: {asset.get('owner', '-')}"
            )
        except Exception as exc:  # 어떤 예외도 밖으로 내보내지 않습니다.
            last_kind = classify_failure(exc)
            last_error = type(exc).__name__
            if last_kind == "fatal":
                break  # 다시 물어도 같은 실패 → 재시도 낭비 없이 즉시 대체 경로로
            # retryable / backoff → 다음 반복에서 재시도

    backup = get_asset_status_backup(asset_id)
    return (
        f"기본 조회가 실패했습니다 (오류: {last_error}, 분류: {last_kind}, 시도 {attempts}회). "
        f"get_asset_status_backup 으로 대체했습니다. 캐시 값이라 현재 상태와 다를 수 있습니다.\n"
        f"{backup}"
    )
    raise NotImplementedError("get_asset_status 를 구현하세요.")


def get_asset_status_backup(asset_id: str) -> str:
    """대체 조회 경로입니다. 캐시된 값이나 축약 정보를 돌려준다고 가정합니다."""
    # TODO: 구현하세요. 실패해도 예외를 던지지 않습니다.
    try:
        asset_id = str(asset_id or "").strip()
        if not asset_id:
            return "get_asset_status_backup: 자산 ID가 비어 있어 캐시를 조회할 수 없습니다."

        # 대체 경로는 외부 시스템(_default_fetch)이 아니라 로컬 캐시(DATA)를 직접 읽는다고 가정합니다.
        assets = DATA.get("assets", {})
        asset = assets.get(asset_id)
        if not asset:
            return (
                f"get_asset_status_backup: '{asset_id}'의 캐시 정보가 없습니다. "
                f"등록된 자산 ID: {', '.join(sorted(assets))}"
            )
        return (
            f"get_asset_status_backup(캐시): {asset_id} 상태: {asset.get('status', '-')}, "
            f"이름: {asset.get('name', '-')}"
        )
    except Exception:
        return "get_asset_status_backup: 캐시 조회 중 오류가 발생했습니다. 담당팀에 확인하세요."
    raise NotImplementedError("get_asset_status_backup 을 구현하세요.")


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

def classify_failure(error: BaseException | str) -> str:
    """실패를 세 종류로 분류합니다. LLM을 호출하지 않습니다.

    반환값은 정확히 다음 셋 중 하나여야 합니다.

      "retryable"  타임아웃, 일시적 연결 오류      -> 짧게 재시도
      "backoff"    요청 한도 초과 (rate limit)     -> 간격을 늘려 재시도
      "fatal"      스키마 오류, 인증 실패, 없는 자원 -> 재시도하지 말고 즉시 대체 경로

    판단 근거는 예외 타입이나 메시지 문자열 어느 쪽을 써도 됩니다.
    """
    # TODO: 구현하세요.
    text = str(error).lower()
    if isinstance(error, BaseException):
        text = f"{type(error).__name__} {text}".lower()

    # 1) 요청 한도 초과 → 간격을 늘려 재시도 (메시지로 판단하므로 가장 먼저 검사)
    if any(k in text for k in ("429", "rate limit", "too many requests", "throttl", "quota")):
        return "backoff"

    # 2) 타임아웃·연결 오류 타입 → 다시 물으면 답이 달라질 수 있음
    if isinstance(error, (TimeoutError, ConnectionError)):
        return "retryable"

    # 3) 없는 자원·스키마 오류·권한 오류 타입 → 몇 번을 물어도 같음
    if isinstance(error, (LookupError, ValueError, PermissionError)):
        return "fatal"

    # 4) 타입으로 못 가른 경우(문자열 입력 등)는 메시지 키워드로 판단
    if any(k in text for k in ("schema", "validation", "not found", "401", "403",
                               "unauthorized", "forbidden", "auth")):
        return "fatal"
    if any(k in text for k in ("timeout", "timed out", "connection", "temporar",
                               "unavailable", "503", "reset")):
        return "retryable"

    # 5) 모르는 실패는 재시도 낭비를 막기 위해 fatal로 봅니다.
    return "fatal"
    raise NotImplementedError("classify_failure 를 구현하세요.")
