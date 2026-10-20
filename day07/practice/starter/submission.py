"""Day 7 — 계측과 누적 평가.

6일간 쌓은 시스템을 측정합니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. run_eval과 compare_eval은 LLM을 호출하지 않습니다.
     '답이 맞았는가'의 판정은 호출자가 넘겨 준 함수가 합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# 판정 결과에 쓰는 상태값입니다. 이 셋만 씁니다.
PASS, FAIL, ERROR = "pass", "fail", "error"


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

class TraceCollector:
    """실행 중 일어난 일을 기록합니다.

    LangChain 콜백 핸들러로도 쓸 수 있게 만들되, 여기서는 최소 인터페이스만 요구합니다.
    trace는 결정적 구조체라서, LLM이 무슨 말을 했든 이벤트 개수와 필드는 확정 검사됩니다.
    """

    def __init__(self) -> None:
        # TODO: 기록을 담을 목록을 준비하세요.
        # 이벤트를 기록 순서대로 쌓아 둘 목록입니다.
        # 순번(seq)은 이 목록의 길이로 매기므로 별도 카운터를 두지 않습니다.
        self._events: list[dict] = []

    def record(self, event: str, name: str, **fields: Any) -> None:
        """이벤트 하나를 기록합니다.

        event: "llm_start" | "llm_end" | "tool_start" | "tool_end" 중 하나
        name : 모델 이름 또는 도구 이름
        fields: latency_s, tokens 등 부가 정보
        """
        # TODO: 구현하세요. 최소한 event, name, 그리고 순번을 남기세요.
        # 핵심 필드(순번·이벤트·이름)를 먼저 둡니다.
        # 순번은 0부터 시작하며, 실행 순서를 그대로 재구성할 수 있게 해 줍니다.
        entry: dict = {"seq": len(self._events), "event": event, "name": name}

        # latency_s, tokens 같은 부가 필드를 덧붙입니다.
        # 부가 필드가 핵심 필드 이름(seq/event/name)과 겹치면 핵심 필드를 지킵니다.
        for key, value in fields.items():
            entry.setdefault(key, value)

        self._events.append(entry)

    @property
    def events(self) -> list[dict]:
        """기록된 이벤트 목록을 반환합니다."""
        # TODO: 구현하세요.
        # 바깥에서 목록이나 항목을 고쳐도 내부 기록이 바뀌지 않도록 사본을 돌려줍니다.
        return [dict(e) for e in self._events]

    def dump(self, path: str | Path) -> None:
        """이벤트를 JSON Lines 파일로 씁니다. 한 줄에 한 이벤트."""
        # TODO: 구현하세요.
        path = Path(path)
        # 저장 위치의 상위 폴더가 없으면 만들어 둡니다.
        path.parent.mkdir(parents=True, exist_ok=True)

        # JSON Lines: 한 줄에 이벤트 하나.
        #  - ensure_ascii=False: 한글 도구 이름 등을 그대로 읽을 수 있게 남깁니다.
        #  - default=str: 직렬화가 안 되는 부가 필드 값도 문자열로 바꿔 기록이 끊기지 않게 합니다.
        with path.open("w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def run_eval(answer_fn: Callable[[str], Any], eval_set: list[dict],
             judge: Callable[[dict, Any], bool] | None = None) -> dict:
    """평가셋을 전부 돌려 유형별 통과율을 냅니다.

    Args:
        answer_fn: 질문 문자열을 받아 답을 돌려주는 함수. 예외가 날 수 있습니다.
        eval_set: [{"id", "question", "type", ...}] 목록
        judge: (문항, 답) -> 통과 여부. None이면 '답이 비어 있지 않으면 통과'로 봅니다.

    Returns:
        {
          "total": 전체 문항 수,
          "passed": 통과 수,
          "by_type": {유형: {"total": n, "passed": m}},
          "failures": [{"id", "type", "status", "detail"}],
        }

    반드시 지킬 것:
      - answer_fn이 예외를 던져도 전체 실행이 멈추면 안 됩니다.
        그 문항은 status를 ERROR로 기록하고 계속 진행합니다.
    """
    # TODO: 구현하세요.

    def _default_judge(_item: dict, answer: Any) -> bool:
        """judge가 없을 때 쓰는 기본 판정: 답이 비어 있지 않으면 통과."""
        if answer is None:
            return False
        if isinstance(answer, str):
            # 공백만 있는 문자열도 '빈 답'으로 봅니다.
            return bool(answer.strip())
        if isinstance(answer, (list, tuple, dict, set)):
            return len(answer) > 0
        # 그 밖의 값(객체 등)은 무언가를 돌려준 것이므로 통과로 봅니다.
        return True

    judge_fn = judge if judge is not None else _default_judge

    total = 0
    passed = 0
    by_type: dict[str, dict[str, int]] = {}
    failures: list[dict] = []

    for item in eval_set:
        item_id = item.get("id")
        item_type = item.get("type") or "unknown"
        question = item.get("question", "")

        total += 1
        # 유형별 집계 칸을 먼저 만들어 둡니다. 예외가 난 문항도 total에는 반드시 들어갑니다.
        bucket = by_type.setdefault(item_type, {"total": 0, "passed": 0})
        bucket["total"] += 1

        # 1) 답 생성 — 여기서 난 예외는 '터진 코드'이므로 ERROR로 기록하고 다음 문항으로 넘어갑니다.
        try:
            answer = answer_fn(question)
        except Exception as exc:  # noqa: BLE001 — 어떤 예외든 전체 실행을 멈추지 않습니다.
            failures.append({
                "id": item_id,
                "type": item_type,
                "status": ERROR,
                "detail": f"answer_fn 예외: {type(exc).__name__}: {exc}",
            })
            continue

        # 2) 판정 — 주입한 judge가 터지는 경우도 같은 이유로 ERROR로 격리합니다.
        try:
            ok = bool(judge_fn(item, answer))
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "id": item_id,
                "type": item_type,
                "status": ERROR,
                "detail": f"judge 예외: {type(exc).__name__}: {exc}",
            })
            continue

        if ok:
            passed += 1
            bucket["passed"] += 1
        else:
            # '틀린 답'은 FAIL. 원인 파악을 돕도록 답의 앞부분을 detail에 남깁니다.
            preview = repr(answer)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            failures.append({
                "id": item_id,
                "type": item_type,
                "status": FAIL,
                "detail": f"판정 불통과. 답: {preview}",
            })

    return {
        "total": total,
        "passed": passed,
        "by_type": by_type,
        "failures": failures,
    }


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

def compare_eval(before: dict, after: dict) -> dict:
    """개선 전후를 비교합니다. LLM을 호출하지 않습니다.

    통과율 총합만 보면 '무엇이 좋아지고 무엇이 깨졌는지'를 알 수 없습니다.
    한 유형을 고치면 다른 유형이 깨지는 일이 실제로 자주 일어납니다.

    Args:
        before, after: run_eval이 반환한 dict

    Returns:
        {
          "fixed":     [before에서 실패했다가 after에서 통과한 문항 id],
          "regressed": [before에서 통과했다가 after에서 실패한 문항 id],
          "delta": after.passed - before.passed,
          "safe": regressed가 비었으면 True,
        }
    """
    # TODO: 구현하세요.
    # run_eval의 결과에는 '실패한 문항 id'만 들어 있습니다.
    # 같은 평가셋을 돌린 두 결과라고 보고, 실패 목록의 차이로 개선·회귀를 가립니다.
    # (FAIL과 ERROR는 모두 '통과하지 못함'으로 같이 취급합니다.)
    before_failed = [f.get("id") for f in before.get("failures", [])]
    after_failed = [f.get("id") for f in after.get("failures", [])]
    before_set, after_set = set(before_failed), set(after_failed)

    # 순서를 원래 목록 순서로 유지해 매번 같은 결과가 나오게 합니다(결정성).
    # fixed    : before에서 실패 → after에서 실패 목록에 없음(= 통과)
    fixed = [i for i in dict.fromkeys(before_failed) if i not in after_set]
    # regressed: before에서 실패 목록에 없음(= 통과) → after에서 실패
    regressed = [i for i in dict.fromkeys(after_failed) if i not in before_set]

    return {
        "fixed": fixed,
        "regressed": regressed,
        "delta": after.get("passed", 0) - before.get("passed", 0),
        # 회귀가 하나라도 있으면 통과 수가 늘었어도 안전하지 않습니다.
        "safe": not regressed,
    }
