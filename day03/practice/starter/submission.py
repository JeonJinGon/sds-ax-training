"""Day 3 — 장애 원인 조사 Agent.

도구 세 개로 장애 원인을 좁혀 갑니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 모델은 함수 안에서 만듭니다. `llm` 인자가 오면 그것을 씁니다.
  3. 도구는 결정적이어야 합니다. 같은 인자에 항상 같은 결과를 돌려주세요.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"

DATA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "ops_fixtures.json").read_text(
        encoding="utf-8"
    )
)


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# TODO: 각 도구의 인자 스키마를 정의하세요.
#class DeployArgs(BaseModel):
#    service: str = Field(..., description="TODO: 어떤 값을 넣어야 하는지 적으세요.")

class DeployArgs(BaseModel):
    service: str = Field(..., description="조회할 서비스 이름. 예: payment-api, auth-service")


class LogsArgs(BaseModel):
    service: str = Field(..., description="로그를 조회할 서비스 이름. 예: payment-api")
    since: str = Field(
        default="",
        description=(
            "로그를 볼 시점. get_deploy_history 결과의 배포 시각(ISO 8601)을 그대로 넣습니다. "
            "예: 2026-08-11T02:58:00+09:00. 비우면 평상시 로그를 봅니다."
        ),
    )


class MetricsArgs(BaseModel):
    service: str = Field(..., description="지표를 조회할 서비스 이름. 예: payment-api")

@tool(args_schema=DeployArgs)
def get_deploy_history(service: str) -> str:
#    """TODO: 이 도구가 무엇을 하고 언제 써야 하는지 30자 이상으로 적으세요.
# 모델이 보는 것은 이 설명문뿐입니다. '언제 쓰지 말아야 하는지'도 적으면 좋습니다.
#    """
    """서비스의 최근 배포 이력(버전, 배포 시각, 배포자)을 최신순으로 조회합니다.
    장애 원인 조사의 첫 단계로, 장애 직전에 배포가 있었는지 확인할 때 씁니다.
    여기서 얻은 배포 시각은 get_logs의 since 인자로 넘깁니다.
    오류율·지연 같은 수치가 궁금할 때는 쓰지 말고 get_metrics를 쓰세요.
    """
    
    # TODO: DATA["deploys"]에서 조회해 사람이 읽을 수 있는 문자열로 반환하세요.
    service = (service or "").strip()
    if not service:
        return "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요."

    deploys = DATA.get("deploys", {}).get(service)
    if not deploys:
        return f"{service}: 배포 이력이 없습니다. 서비스 이름을 확인하세요."

    lines = [f"[{service} 최근 배포 이력 — 최신순]"]
    for d in deploys:
        lines.append(f"- {d['version']} / 배포 시각 {d['at']} / 배포자 {d['by']}")
    lines.append("배포 직후 로그를 보려면 get_logs의 since에 위 배포 시각을 그대로 넣으세요.")
    return "\n".join(lines)

    raise NotImplementedError("get_deploy_history 를 구현하세요.")


# TODO: get_logs 를 만드세요.
#   - 인자: service, since (배포 시각). since가 있으면 그 시점 로그를 봅니다.
#   - 이 도구는 get_deploy_history 결과가 있어야 제대로 쓸 수 있습니다.
@tool(args_schema=LogsArgs)
def get_logs(service: str, since: str = "") -> str:
    """서비스의 애플리케이션 로그(에러·경고)를 조회합니다.

    배포 직후 무슨 에러가 찍혔는지 확인할 때 씁니다. 먼저 get_deploy_history로
    배포 시각을 알아낸 뒤 그 값을 since에 넣어야 해당 구간 로그가 나옵니다.
    since 없이 부르면 평상시 로그만 보이므로 원인 조사에는 부족합니다.
    """
    service = (service or "").strip()
    if not service:
        return "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요."

    logs = DATA.get("logs", {})
    since = (since or "").strip()

    # since가 있으면 "서비스@배포시각" 키로 그 시점 로그를 찾습니다.
    if since:
        key = f"{service}@{since}"
        if key in logs:
            return f"[{service} 로그 — {since} 배포 시점]\n{logs[key]}"

    # since가 없거나 그 시점 로그가 없으면 평상시 로그를 보여 줍니다.
    default = logs.get(f"{service}@default")
    if default is None:
        return f"{service}: 로그가 없습니다. 서비스 이름을 확인하세요."
    note = f"{since} 시점 로그는 없어 평상시 로그를 보여 줍니다.\n" if since else ""
    return f"{note}[{service} 평상시 로그]\n{default}"
# TODO: get_metrics 를 만드세요.
#   - 인자: service
#   - DATA["metrics"]에서 조회합니다.
@tool(args_schema=MetricsArgs)
def get_metrics(service: str) -> str:
    """서비스의 현재 운영 지표(5분 오류율, p99 지연, CPU 사용률)를 조회합니다.

    장애의 규모와 영향을 수치로 확인할 때 씁니다. 원인이 된 배포나 에러 메시지는
    알려 주지 않으므로, 원인을 찾을 때는 get_deploy_history와 get_logs를 쓰세요.
    """
    service = (service or "").strip()
    if not service:
        return "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요."

    m = DATA.get("metrics", {}).get(service)
    if m is None:
        return f"{service}: 해당 서비스의 지표가 없습니다. 서비스 이름을 확인하세요."

    return (
        f"[{service} 지표]\n"
        f"- 5분 오류율: {m['error_rate_5m'] * 100:.1f}%\n"
        f"- p99 지연: {m['p99_latency_ms']}ms\n"
        f"- CPU 사용률: {m['cpu'] * 100:.0f}%"
    )

# TODO: 위에서 만든 도구들을 여기에 모으세요.
TOOLS: list = [get_deploy_history, get_logs, get_metrics]


def build_agent(llm=None):
    """도구를 쓸 수 있는 Agent 그래프를 만들어 컴파일해 반환합니다.

    반드시 지킬 것:
      - agent 노드와 tools 노드 사이에 순환이 있어야 합니다.
      - 탈출 조건이 있어야 합니다. 없으면 무한 루프입니다.

    create_react_agent를 써도 되고 StateGraph로 직접 만들어도 됩니다.
    심화 파트를 구현했다면 should_stop 판정을 그래프에 연결하세요.
    """
    
    llm = llm or _default_llm()
    # TODO: 구현하세요.

    from langchain.agents import create_agent
    from langchain.agents.middleware import after_model, before_model

    # 모델에게 줄 조사 규칙. 연쇄(배포 시각 → since)를 유도하는 문장이 핵심입니다.
    system_rule = (
        "당신은 장애 원인 조사 담당자입니다. 다음 순서로 조사하세요.\n"
        "1) get_deploy_history로 최근 배포와 배포 시각을 확인합니다.\n"
        "2) 그 배포 시각을 get_logs의 since에 그대로 넣어 해당 구간 로그를 봅니다.\n"
        "3) get_metrics로 영향 규모를 확인합니다.\n"
        "도구가 필요 없는 일반 질문에는 도구를 부르지 말고 바로 답하세요. "
        "같은 도구를 같은 인자로 다시 부르지 마세요."
    )

    def _tool_count(messages):
        """지금까지 실제로 실행된 도구 횟수 = ToolMessage 개수."""
        return sum(1 for m in messages if getattr(m, "type", "") == "tool")

    # ── 훅 1: 모델 응답 직후 (도구 실행 전) ─────────────────────────
    # 모델이 요청한 도구 호출이 예산을 넘기거나 중복이면 실행하지 않고 종료합니다.
    @after_model(can_jump_to=["end"])
    def block_bad_calls(state, runtime):
        messages = state["messages"]
        pending = getattr(messages[-1], "tool_calls", None) or []
        if not pending:
            return None  # 도구 호출이 없으면 create_agent가 알아서 종료합니다.

        made = _tool_count(messages)
        # 탈출 조건: 이번 호출까지 실행하면 예산 초과 → 실행 전에 차단
        if made + len(pending) > MAX_TOOL_CALLS:
            return {"jump_to": "end"}
        # 탈출 조건: 같은 도구·같은 인자 재호출 → 실행 전에 차단
        stop, reason = should_stop(messages, made)
        if stop and reason == "중복 호출":
            return {"jump_to": "end"}
        return None  # 통과 → tools 노드로 진행

    # ── 훅 2: 모델 호출 직전 (도구 실행 후, 다음 판단 전) ────────────
    # 예산을 다 썼거나 세 도구를 모두 써서 근거가 충분하면 더 묻지 않고 종료합니다.
    @before_model(can_jump_to=["end"])
    def stop_when_enough(state, runtime):
        messages = state["messages"]
        stop, _ = should_stop(messages, _tool_count(messages))
        return {"jump_to": "end"} if stop else None

    # create_agent가 model ↔ tools 순환을 만들고, 미들웨어가 그 사이에서 탈출 조건을 검사합니다.
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=system_rule,
        middleware=[stop_when_enough, block_bad_calls],
    )

    class _GuardedModel:
        """실제 모델을 감싸, 응답이 나올 때마다 _guard 검사를 끼워 넣는 래퍼입니다."""

        def __init__(self, inner):
            self.inner = inner

        def bind_tools(self, tools, **kwargs):
            # create_agent가 도구를 묶을 때도 래퍼가 유지되도록 다시 감쌉니다.
            return _GuardedModel(self.inner.bind_tools(tools, **kwargs))

        def invoke(self, messages, config=None, **kwargs):
            return _guard(messages, self.inner.invoke(messages, config, **kwargs))

        async def ainvoke(self, messages, config=None, **kwargs):
            return _guard(messages, await self.inner.ainvoke(messages, config, **kwargs))

        def __getattr__(self, name):
            # 그 밖의 속성 접근은 실제 모델에 그대로 위임합니다.
            return getattr(self.inner, name)

    return create_agent(model=_GuardedModel(llm), tools=TOOLS, system_prompt=system_rule)

    raise NotImplementedError("build_agent 를 구현하세요.")


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# TODO: 한 번의 조사에서 허용할 도구 호출 상한 (3 이상 8 이하)
MAX_TOOL_CALLS = 5


def should_stop(messages: list, tool_calls_made: int) -> tuple[bool, str]:
    """더 조사할지 멈출지 결정합니다. LLM을 호출하지 않습니다.

    판정 규칙:
      - tool_calls_made >= MAX_TOOL_CALLS        -> (True, "예산 초과")
      - 같은 도구를 같은 인자로 2회 이상 호출함  -> (True, "중복 호출")
      - 세 도구를 모두 한 번 이상 호출함         -> (True, "근거 충분")
      - 그 외                                    -> (False, 아무 문자열)

    Args:
        messages: 지금까지의 메시지 목록. 도구 호출 기록이 들어 있습니다.
        tool_calls_made: 지금까지 실행한 도구 호출 횟수

    Returns:
        (중단할지 여부, 이유 문자열)
    """
    # TODO: 구현하세요.
    # 규칙 1: 예산 초과
    if tool_calls_made >= MAX_TOOL_CALLS:
        return True, "예산 초과"

    required = {"get_deploy_history", "get_logs", "get_metrics"}
    seen_calls = set()   # (도구 이름, 인자 JSON) 조합
    used_tools = set()   # 한 번이라도 부른 도구 이름

    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc["name"] if isinstance(tc, dict) else tc.name
            args = tc.get("args", {}) if isinstance(tc, dict) else tc.args
            # 인자 dict는 set에 넣을 수 없으므로, 키 순서를 고정한 JSON 문자열로 바꿔 비교합니다.
            key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))

            # 규칙 2: 같은 도구 + 같은 인자를 두 번째로 만남
            if key in seen_calls:
                return True, "중복 호출"
            seen_calls.add(key)
            used_tools.add(name)

    # 규칙 3: 세 도구를 모두 한 번 이상 사용
    if required <= used_tools:
        return True, "근거 충분"

    # 규칙 4: 그 외 → 계속 조사
    return False, "조사 계속"
    raise NotImplementedError("should_stop 을 구현하세요.")
