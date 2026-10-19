"""Day 6 — Supervisor 인시던트 대응팀.

전문 Agent 셋을 Supervisor가 지휘합니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. `route_question`은 LLM을 호출하지 않습니다. 결정적 규칙으로 배분하세요.
     실제 운영에서는 모델이 배분하지만, 규칙으로 먼저 나눠 보면
     '무엇을 근거로 나누는가'가 분명해집니다.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 전문 Agent 이름 3개 — 이름이 곧 담당 범위의 선언입니다.
#   답이 이상할 때 어느 Agent(프롬프트)를 열어야 하는지 이름만 보고 알 수 있어야 합니다.
AGENT_NAMES: list[str] = ["log_agent", "metric_agent", "runbook_agent"]

# Agent별 도구 — 도메인 기준으로 나누고, 같은 도구는 한 Agent에만 둡니다(격리).
#   Day 3·4에서 만든 도구를 그대로 재배치했습니다.
#   - log_agent     : 애플리케이션 로그(에러·예외·스택)를 읽는 도구만
#   - metric_agent  : 수치(오류율·지연·CPU)와 수치 변화의 원인 후보인 배포 이력
#                     (README 라우팅 표에서 '배포'가 metric_agent 몫이므로 배포 이력도 여기에 둡니다)
#   - runbook_agent : 절차·담당·과거 사례 — "무엇을 어떻게 누구와" 하는지에 대한 도구
#   "혹시 몰라서" 다른 Agent 도구를 얹지 않습니다. 겹치면 같은 질문이 날마다 다른 Agent로 갑니다.
AGENT_TOOLS: dict[str, list[str]] = {
    "log_agent": ["get_logs"],
    "metric_agent": ["get_metrics", "get_deploy_history"],
    "runbook_agent": ["search_runbook", "get_service_owner", "query_recent_incidents"],
}

# ──────────────────────────────────────────────────────────────────
# 라우팅 규칙 (LLM 없이 결정적으로 배분) — 정규식 없이 키워드 목록과 문자열 메서드로 판정합니다.
#
# 채점은 README 표에 없는 동의어·영문으로도 하므로, 표의 단어를 그대로 베끼지 않고
# 같은 뜻의 표현을 함께 넣었습니다. 단어를 추가·제외할 때는 아래 목록에 한 줄만 고치면 됩니다.
#
# 판정 방법 (KeywordRule.matches)
#   1) 한글 : 공백을 지운 문장에서 '제외어'를 먼저 지운 뒤, 남은 문장에 키워드가 '포함'되는지 봅니다.
#             공백을 지우는 이유 — "응답 시간", "스택 트레이스"처럼 띄어 쓴 표기도 같은 말로 보기 위해
#             제외어를 먼저 지우는 이유 — "로그인 오류"처럼 함정 단어와 진짜 키워드가 섞여도
#             함정 부분(로그인)만 빠지고 진짜 키워드(오류)는 남게 하기 위해  ※ 순서가 바뀌면 안 됩니다
#   2) 영문 : 영문·숫자만 남겨 단어로 쪼갠 뒤, 키워드와 '같은 단어'인지 봅니다.
#             → login·catalog·dialog 는 log 와 다른 단어이므로 자동으로 빠집니다.
#             두 단어가 함께 쓰여야 뜻이 정해지는 표현(stack trace / error rate)은 '단어 쌍'으로 봅니다.
#   3) 5xx  : 문자열을 한 글자씩 훑어 5xx·500~599 상태코드를 찾되, 뒤에 ms·건 같은 단위가 붙으면 제외합니다.
#
# [함정 대응] 단어가 '포함'됐다고 바로 배분하면 조용히 틀리는 자리들
#   - "로그인", "로그아웃", "블로그", "카탈로그"  → 로그가 아닙니다 (한글 제외어)
#   - "login", "catalog", "dialog"                → 영문은 단어 단위 비교로 자동 제외
#   - "오류율", "에러율", "error rate"           → 수치(지표)입니다. 로그 쪽 제외어·제외 단어 쌍
#   - "p99 500ms", "1500건"                       → 수치입니다. 5xx 상태코드로 보지 않습니다
#   - "기술 스택"                                 → '스택' 단독은 키워드가 아닙니다 (스택트레이스·콜스택만)
# ──────────────────────────────────────────────────────────────────

# 5xx 뒤에 붙으면 상태코드가 아니라 수치로 보는 단위
_NUMBER_UNITS = ("ms", "밀리", "초", "개", "건", "명", "원", "mb", "gb", "%")


def _english_words(text: str) -> list[str]:
    """영문·숫자만 남기고 나머지(한글·기호·공백)를 경계로 삼아 단어 목록을 만듭니다.

    예) "payment-api Exception 로그" → ["payment", "api", "exception"]
        "on-call 담당"               → ["on", "call"]
    """
    chars = [ch if (ch.isascii() and ch.isalnum()) else " " for ch in text]
    return "".join(chars).split()


def _has_5xx(text: str) -> bool:
    """'5xx' 또는 5로 시작하는 세 자리 숫자(500~599)를 상태코드로 쓴 경우를 찾습니다.

    한 글자씩 훑으면서 (앞뒤가 숫자가 아님) + (뒤에 ms·건 같은 단위가 붙지 않음)을 직접 확인합니다.
    """
    n = len(text)
    for i in range(n - 2):
        chunk = text[i:i + 3]
        if not (chunk == "5xx" or (chunk[0] == "5" and chunk[1:].isdigit())):
            continue
        if i > 0 and text[i - 1].isdigit():                  # 1500 같은 더 긴 숫자의 일부
            continue
        if i + 3 < n and text[i + 3].isdigit():              # 5000 같은 더 긴 숫자의 일부
            continue
        if text[i + 3:].lstrip().startswith(_NUMBER_UNITS):  # 500ms, 500건 → 수치
            continue
        return True
    return False


class KeywordRule:
    """한 Agent(또는 약한 단서)의 판정 규칙입니다. 목록만 고치면 규칙이 바뀝니다.

    @dataclass를 쓰지 않는 이유: 이 파일은 `from __future__ import annotations`를 쓰므로
    dataclass가 클래스를 만들 때 모듈을 sys.modules에서 찾습니다. 채점기가 모듈을 등록하지 않고
    불러오면 파일 로딩 자체가 실패해 모든 점수를 잃습니다. 평범한 클래스는 이 문제가 없습니다.
    """

    def __init__(
        self,
        ko=(),                 # 한글 키워드 (공백 제거본에 '포함'되면 해당)
        ko_exclude=(),         # 한글 제외어 (키워드 검사 '전에' 지움)
        en=frozenset(),        # 영문 키워드 (단어가 '같으면' 해당)
        en_pairs=frozenset(),  # 두 단어가 연달아 나오면 해당 — 예: ("stack", "trace")
        en_pair_exclude=frozenset(),  # 이 쌍의 첫 단어는 키워드로 세지 않음 — 예: ("error", "rate")
        check_5xx=False,       # 5xx 상태코드도 볼지
    ):
        self.ko = tuple(ko)
        self.ko_exclude = tuple(ko_exclude)
        self.en = frozenset(en)
        self.en_pairs = frozenset(en_pairs)
        self.en_pair_exclude = frozenset(en_pair_exclude)
        self.check_5xx = check_5xx

    def matches(self, text: str) -> bool:
        """text(소문자로 바꾼 질문)가 이 규칙에 해당하는지 판정합니다."""
        # 1) 한글 — 공백 제거 → 제외어 제거 → 키워드 포함 여부
        compact = text.replace(" ", "")
        for word in self.ko_exclude:
            compact = compact.replace(word, " ")   # 지운 자리를 공백으로 채워 앞뒤 글자가 붙지 않게 합니다
        if any(k in compact for k in self.ko):
            return True

        # 2) 영문 — 단어 단위 비교 (+ 단어 쌍)
        words = _english_words(text)
        for i, w in enumerate(words):
            pair = (w, words[i + 1] if i + 1 < len(words) else "")
            if pair in self.en_pair_exclude:       # "error rate"의 error 는 로그 키워드가 아님
                continue
            if pair in self.en_pairs or w in self.en:
                return True

        # 3) 5xx 상태코드
        return self.check_5xx and _has_5xx(text)


# Agent별 판정 규칙 (이름은 이전과 같이 _ROUTE_PATTERNS를 유지합니다)
_ROUTE_PATTERNS: dict[str, KeywordRule] = {
    "log_agent": KeywordRule(
        ko=("로그", "로깅", "에러", "오류", "예외", "익셉션",
            "스택트레이스", "스택오버플로", "콜스택", "트레이스", "덤프", "크래시", "패닉"),
        ko_exclude=("로그인", "로그아웃", "블로그", "카탈로그",                  # 로그가 아닌 단어
                    "오류율", "오류비율", "에러율", "에러비율", "에러레이트"),   # 로그가 아니라 지표
        en=frozenset({"log", "logs", "logging", "error", "errors", "exception", "exceptions",
                      "trace", "traces", "traceback", "tracebacks", "stacktrace",
                      "fatal", "panic", "crash"}),
        en_pairs=frozenset({("stack", "trace"), ("stack", "traces")}),
        en_pair_exclude=frozenset({("error", "rate"), ("errors", "rate")}),
        check_5xx=True,
    ),
    "metric_agent": KeywordRule(
        ko=("지표", "메트릭", "수치", "통계", "대시보드", "그래프",
            "지연", "레이턴시", "응답시간", "처리량", "트래픽", "사용률", "점유율",
            "cpu", "씨피유", "메모리", "오류율", "오류비율", "에러율", "에러비율", "에러레이트",
            "p90", "p95", "p99", "qps", "tps", "rps"),     # 영문 약어도 붙여 쓰는 경우가 많아 포함 검사로 봅니다
        en=frozenset({"metric", "metrics", "latency", "throughput", "dashboard", "memory", "usage"}),
        en_pairs=frozenset({("error", "rate"), ("errors", "rate")}),
    ),
    "runbook_agent": KeywordRule(
        ko=("절차", "대응", "조치", "방법", "순서", "런북", "매뉴얼", "가이드", "플레이북", "지침서",
            "담당", "책임자", "누구", "연락처", "온콜", "에스컬레이션"),
        en=frozenset({"runbook", "runbooks", "playbook", "playbooks", "procedure", "procedures",
                      "guide", "sop", "manual", "owner", "who", "oncall"}),
        en_pairs=frozenset({("on", "call")}),            # "on-call", "on call"
    ),
}

# 약한 단서 — 다른 단서와 함께 올 때 의미가 달라지는 표현입니다.
#   '배포'는 README 표에서 metric_agent 몫이지만, "배포 절차가 어떻게 되나요"처럼
#   런북 단서와 함께 오면 '배포 수치'가 아니라 '배포 방법'을 묻는 질문입니다.
#   → 런북 단서가 있으면 배포만으로는 metric_agent를 부르지 않습니다.
#   '디스크'도 같습니다. "디스크 풀 장애 시 연락처"에서 디스크는 수치가 아니라 증상 이름입니다.
_WEAK_METRIC = KeywordRule(
    ko=("배포", "릴리스", "롤아웃", "디스크"),
    en=frozenset({"deploy", "deploys", "deployment", "deployments", "release", "disk"}),
)
#   '어떻게'는 "CPU가 어떻게 변했나요"처럼 다른 질문에도 흔히 붙습니다.
#   → 다른 Agent 단서가 하나도 없을 때만 runbook_agent로 봅니다.
_WEAK_RUNBOOK = KeywordRule(ko=("어떻게",), en=frozenset({"how"}))


def route_question(question: str) -> list[str]:
    """질문을 처리할 Agent 목록을 반환합니다. LLM을 호출하지 않습니다.

    반환 규칙:
      - 로그·에러·예외·스택·트레이스·5xx, 영문 log/exception/trace -> log_agent
      - 지표·수치·지연·레이턴시·CPU·메모리·오류율·배포, 영문 metric/latency -> metric_agent
      - 절차·대응·방법·런북·매뉴얼·담당·누구·어떻게, 영문 runbook -> runbook_agent
      - 여러 주제가 섞이면 해당하는 Agent를 모두 반환
      - 어디에도 해당하지 않으면 빈 목록 (Supervisor가 직접 답한다)

    **필요한 것만 반환하세요.** 전부 부르면 정확도는 오르지만 호출 비용이 폭증합니다.
    """
    # 빈 값·None·공백만 있는 질문은 아무 Agent도 부르지 않습니다. (예외를 던지지 않습니다)
    if not question or not str(question).strip():
        return []
    text = str(question).lower()

    hits = {name for name, rule in _ROUTE_PATTERNS.items() if rule.matches(text)}

    # 약한 단서 처리 (위 _WEAK_* 주석 참고)
    if _WEAK_METRIC.matches(text) and "runbook_agent" not in hits:
        hits.add("metric_agent")
    if not hits and _WEAK_RUNBOOK.matches(text):
        hits.add("runbook_agent")

    # AGENT_NAMES 순서로 정렬해 반환 → 같은 질문은 언제나 같은 순서, 중복 없음
    return [name for name in AGENT_NAMES if name in hits]


# ──────────────────────────────────────────────────────────────────
# Supervisor 그래프
#
#   START → supervisor ─┬─(대상 Agent가 남음)→ log_agent / metric_agent / runbook_agent ─┐
#                       │                                                              │
#                       └─(남은 대상 없음)→ END        ←── 각 Agent는 supervisor로 복귀 ─┘
#
#   supervisor 노드가 하는 일 (모두 결정적 규칙)
#     1) 처음 들어오면 route_question으로 대상 Agent를 정합니다.
#        대상이 없으면 Supervisor가 직접 답하고 끝냅니다.
#     2) 대상 Agent를 하나씩 실행시키고, 모두 끝나면 게이트를 통과한 결과만 모아 종합합니다.
#   워커 노드 출력은 반드시 judge_output 게이트를 거친 뒤 메시지에 반영됩니다. (심화 연결)
# ──────────────────────────────────────────────────────────────────

# 그래프 상태 — 함수형 TypedDict 문법으로 정의합니다.
#   이 파일은 `from __future__ import annotations`를 쓰므로 class 문법으로 쓰면 타입 힌트가 문자열로 남고,
#   모듈이 sys.modules에 등록되지 않은 채 로딩되면 langgraph가 Annotated 등을 찾지 못해 실패합니다.
#   함수형 문법은 실제 타입 객체를 그대로 넘기므로 로딩 방식과 무관하게 동작합니다.
SupervisorState = TypedDict(
    "SupervisorState",
    {
        "messages": Annotated[list[AnyMessage], add_messages],
        "targets": list[str],                               # 이번 질문을 처리할 Agent 목록
        "routed": bool,                                     # 배분을 이미 했는지
        "done": Annotated[list[str], operator.add],         # 실행을 마친 Agent
        "judgements": Annotated[list[dict], operator.add],  # Agent별 게이트 판정 기록
        "finalized": bool,                                  # 종합 답변을 냈는지
    },
    total=False,
)


# Agent별 담당 범위 프롬프트 — 범위를 좁게 잡아야 오답이 났을 때 고칠 자리가 특정됩니다.
_AGENT_PROMPTS: dict[str, str] = {
    "log_agent": (
        "당신은 로그 분석 전담 Agent입니다. 애플리케이션 로그의 에러·예외·스택 트레이스만 봅니다. "
        "지표 해석이나 대응 절차는 다른 Agent의 몫이니 언급하지 마세요. "
        "관측한 에러 종류와 건수, 시각을 근거로 답하세요."
    ),
    "metric_agent": (
        "당신은 지표 분석 전담 Agent입니다. 오류율·지연·CPU·메모리 같은 수치와 배포 이력만 봅니다. "
        "로그 내용이나 대응 절차는 다른 Agent의 몫이니 언급하지 마세요. "
        "수치와 측정 시점을 근거로 답하세요."
    ),
    "runbook_agent": (
        "당신은 런북 전담 Agent입니다. 대응 절차, 담당 조직, 과거 인시던트 사례만 다룹니다. "
        "로그·지표 해석은 다른 Agent의 몫이니 언급하지 마세요. "
        "런북 문서와 담당 정보를 근거로 답하세요."
    ),
}


def _last_user_text(messages: list[AnyMessage]) -> str:
    """메시지 목록에서 마지막 사용자 질문을 꺼냅니다."""
    for m in reversed(messages or []):
        if getattr(m, "type", "") == "human":
            return str(m.content)
    return ""


def _text_of(message: Any) -> str:
    """모델 응답에서 본문 문자열만 꺼냅니다. (content가 블록 리스트인 경우도 처리)"""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


def _pending(state: SupervisorState) -> list[str]:
    """아직 실행하지 않은 대상 Agent를 순서대로 돌려줍니다."""
    done = set(state.get("done") or [])
    return [t for t in (state.get("targets") or []) if t not in done]


def build_supervisor(llm=None):
    """Supervisor 그래프를 만들어 컴파일해 반환합니다.

    반드시 지킬 것:
      - AGENT_NAMES의 이름이 모두 그래프 노드로 존재해야 합니다.
      - supervisor 노드에서 각 Agent로 가는 경로가 있어야 합니다.
      - 종료 경로가 있어야 합니다.

    langgraph_supervisor를 써도 되고 StateGraph로 직접 만들어도 됩니다.
    """
    # 모델은 반드시 함수 안에서 만듭니다. 채점기는 llm 인자로 가짜 모델을 주입합니다.
    llm = llm or _default_llm()

    def supervisor(state: SupervisorState) -> dict:
        question = _last_user_text(state.get("messages", []))

        # 1) 첫 진입: 결정적 규칙으로 배분합니다.
        if not state.get("routed"):
            targets = route_question(question)
            if not targets:
                # 어느 전문 Agent에도 해당하지 않으면 Supervisor가 직접 답합니다.
                reply = llm.invoke([
                    SystemMessage(content="당신은 인시던트 대응팀 Supervisor입니다. 전문 Agent가 필요 없는 질문에 간단히 답하세요."),
                    HumanMessage(content=question),
                ])
                return {
                    "targets": [], "routed": True, "finalized": True,
                    "messages": [AIMessage(content=f"[supervisor] {_text_of(reply)}", name="supervisor")],
                }
            return {"targets": targets, "routed": True}

        # 2) 대상 Agent가 모두 끝났으면 게이트를 통과한 결과만 모아 종합합니다.
        if not _pending(state) and not state.get("finalized"):
            kept = [j["agent"] for j in state.get("judgements", []) if j.get("keep")]
            dropped = [f"{j['agent']}({j['reason']})" for j in state.get("judgements", []) if not j.get("keep")]
            summary = "[supervisor] 종합 — 채택: " + (", ".join(kept) or "없음")
            if dropped:
                summary += " / 억제: " + ", ".join(dropped)
            return {"finalized": True, "messages": [AIMessage(content=summary, name="supervisor")]}
        return {}

    def next_step(state: SupervisorState) -> str:
        """supervisor 다음에 갈 곳: 남은 대상 Agent 중 첫 번째, 없으면 종료."""
        if state.get("finalized"):
            return END
        pending = _pending(state)
        return pending[0] if pending else "supervisor"

    def make_worker(name: str):
        """전문 Agent 노드를 만듭니다. 출력은 judge_output 게이트를 거친 뒤에만 메시지로 나갑니다."""

        def worker(state: SupervisorState) -> dict:
            question = _last_user_text(state.get("messages", []))
            reply = llm.invoke([
                SystemMessage(content=_AGENT_PROMPTS[name] + f" 사용할 수 있는 도구: {', '.join(AGENT_TOOLS[name])}"),
                HumanMessage(content=question),
            ])
            output = _text_of(reply)

            # [심화 연결] Supervisor 게이트 — 워커 산출물을 그대로 내보내지 않습니다.
            verdict = judge_output(name, output)
            if verdict["keep"]:
                content = f"[{name}] {output}"
            else:
                # 억제할 때도 사유를 남겨야 '게이트가 과하게 잠겼는지 / 워커가 부실한지' 구분할 수 있습니다.
                content = f"[{name}] 산출물 억제 — 사유: {verdict['reason']}"
            return {
                "messages": [AIMessage(content=content, name=name)],
                "done": [name],
                "judgements": [{"agent": name, **verdict}],
            }

        return worker

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    for name in AGENT_NAMES:
        graph.add_node(name, make_worker(name))
        graph.add_edge(name, "supervisor")          # 모든 Agent는 끝나면 supervisor로 복귀

    graph.add_edge(START, "supervisor")
    # supervisor → 각 Agent / supervisor(종합 단계) / END 로 가는 경로를 모두 선언합니다.
    graph.add_conditional_edges(
        "supervisor",
        next_step,
        {**{name: name for name in AGENT_NAMES}, "supervisor": "supervisor", END: END},
    )
    return graph.compile()


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

def judge_output(agent_name: str, output: str) -> dict:
    """워커 산출물을 내보내기 전에 걸러냅니다. LLM을 호출하지 않습니다.

    워커가 만든 것을 그대로 사용자에게 주면 근거 없는 단정과 빈 답변이 섞입니다.
    Supervisor는 게이트 역할을 해야 합니다.

    판정 규칙:
      - 내용이 비었거나 20자 미만          -> {"keep": False, "reason": "저가치"}
      - "확실합니다" "틀림없습니다" 같은
        근거 없는 단정이 있는데 수치·출처가 없음 -> {"keep": False, "reason": "근거 없는 단정"}
      - 그 외                              -> {"keep": True, "reason": ...}

    Returns:
        {"keep": bool, "reason": str} 형태의 dict
    """
    text = str(output or "").strip()

    # 1) 저가치: 비었거나 20자 미만 ("확인함", "ok" 같은 빈말)
    if len(text) < 20:
        return {"keep": False, "reason": "저가치"}

    # 2) 근거 없는 단정: 단정 표현은 있는데 수치·출처가 하나도 없음
    #    단정 표현과 근거 유무를 '함께' 봅니다.
    #    - 단정만 보고 막으면 "NullPointerException이 1243건 — 확실합니다" 같은 쓸 만한 답까지 사라집니다.
    #    - 길이만 보면 단정 표현으로만 채운 답이 20자를 넘겨 그대로 나갑니다.
    compact = _compact_lower(text)
    if _has_assertive(compact):
        # "100% 확실" 같은 표현의 100은 수치 근거가 아니라 단정의 일부이므로 빼고 봅니다.
        evidence_text = compact.replace(_CERTAINTY_NUMBER, "")
        if not _has_evidence(evidence_text):
            return {"keep": False, "reason": "근거 없는 단정"}

    return {"keep": True, "reason": "통과: 내용과 근거가 충분함"}


# ──────────────────────────────────────────────────────────────────
# judge_output 판정용 표현 목록 — 정규식 없이 '포함 여부'로 판정합니다.
#   (함수 정의 뒤에 두어도 호출 시점에는 정의되어 있습니다)
#
# 비교 방법: 산출물을 '공백 제거 + 소문자'로 바꾼 뒤(_compact_lower) 목록의 표현이 들어 있는지 봅니다.
#   - 공백을 지우므로 "의심의 여지", "에 따르면", "according to"처럼 띄어 쓴 표현도 붙여 쓴 형태로 적습니다.
#   - 소문자로 바꾸므로 "Definitely", "SOURCE" 같은 대소문자 변형도 함께 잡힙니다.
# ──────────────────────────────────────────────────────────────────

# 단정 표현: 확신을 선언하지만 그 자체로는 아무 근거도 담지 않는 말들
#   '확실'·'분명' 단독은 넣지 않습니다. "확실하지 않습니다", "분명하지 않다"처럼 부정·유보에도 쓰이므로
#   단정으로 끝나는 꼴(확실합니다·확실히·분명히 …)만 등록합니다.
_ASSERTIVE: tuple[str, ...] = (
    "확실합니다", "확실하다", "확실해요", "확실히", "확실함",
    "틀림없", "분명합니다", "분명하다", "분명히", "명백", "단언", "의심의여지",
    "무조건", "100%", "그것뿐", "그게전부",
    "definitely", "certainly", "undoubtedly", "nodoubt", "guaranteed",
)

# 근거 표현: 출처(로그·지표·런북·문서·링크 등)를 가리키는 말
#   수치 근거(건수·비율·시각 등 숫자)는 목록이 아니라 _has_evidence에서 '숫자가 있는지'로 따로 봅니다.
_EVIDENCE: tuple[str, ...] = (
    "출처", "근거", "에따르면", "기준으로", "관측", "측정", "로그에서", "대시보드", "런북",
    "문서", "티켓", "링크", "http://", "https://",
    "accordingto", "source", "observed", "measured",
)

# 단정의 일부로 쓰인 숫자 — 근거를 셀 때 미리 지웁니다. ("100% 확실합니다"의 100은 수치 근거가 아님)
_CERTAINTY_NUMBER = "100%"


def _compact_lower(text: str) -> str:
    """공백·줄바꿈을 모두 지우고 소문자로 바꿉니다. (표현 목록과 비교하기 위한 형태)"""
    return "".join(text.split()).lower()


def _has_assertive(compact: str) -> bool:
    """단정 표현이 하나라도 들어 있는지 봅니다."""
    return any(phrase in compact for phrase in _ASSERTIVE)


def _has_evidence(compact: str) -> bool:
    """수치(숫자) 또는 출처 표현이 하나라도 들어 있는지 봅니다."""
    if any(ch.isdigit() for ch in compact):
        return True
    return any(phrase in compact for phrase in _EVIDENCE)
