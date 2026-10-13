"""Day 2 — 런북 RAG와 평가셋.

문서를 인덱싱하고, 질문에 근거와 함께 답합니다.

[중요] 이 파일의 규칙
  1. 함수 이름과 인자를 바꾸지 마세요.
  2. 모델·임베딩은 반드시 함수 **안에서** 만듭니다.
  3. `llm` 인자가 들어오면 그것을 씁니다. 채점기가 가짜 모델을 넣습니다.
  4. `build_retriever`와 `build_rag_chain`은 **주어진 vectorstore/retriever를 그대로 씁니다.**
     함수 안에서 새로 인덱싱하지 마세요.
"""

from __future__ import annotations

from typing import Any

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"

PERSIST_DIR = "./chroma_db"
COLLECTION = "sds_policies"


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def _default_embeddings():
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def format_docs(docs) -> str:
    """검색된 문서를 프롬프트에 넣을 문자열로 만듭니다."""
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}" for d in docs
    )


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

def build_chunks(doc_paths: list[str]) -> list:
    """문서를 읽어 청크 Document 리스트를 만듭니다.

    반드시 지킬 것:
      - 모든 청크 metadata에 source / department / updated_at 세 키가 있어야 합니다.
      - 메타데이터는 **분할하기 전에** 붙여야 상속됩니다.
      - 고정 길이로 자르지 말고 RecursiveCharacterTextSplitter 또는
        MarkdownHeaderTextSplitter를 쓰세요.

    Args:
        doc_paths: 읽을 마크다운 파일 경로 목록

    Returns:
        langchain_core.documents.Document 의 리스트
    """
    # TODO: 구현하세요.
   
    from langchain_text_splitters import MarkdownHeaderTextSplitter

   
    meta_map = {
        "leave_policy.md": {"department": "people", "updated_at": "2026-03-02"},
        "oncall_policy.md": {"department": "sre", "updated_at": "2026-06-30"},
        "incident_runbook.md": {"department": "sre", "updated_at": "2026-06-30"},
    }

    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
    )

    chunks = []
    for path in doc_paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        name = str(path).replace("\\", "/").split("/")[-1]
        # 매핑에 없는 문서가 들어와도 세 키는 반드시 존재하도록 기본값을 둡니다.
        extra = meta_map.get(name, {"department": "unknown", "updated_at": "unknown"})
        base_meta = {"source": name, **extra}

        for doc in splitter.split_text(text):
            doc.metadata = {**base_meta, **doc.metadata}
            chunks.append(doc)

    return chunks

    raise NotImplementedError("build_chunks 를 구현하세요.")


def build_retriever(vectorstore):
    """주어진 벡터저장소에서 retriever를 만듭니다.

    여기서 새로 인덱싱하지 마세요. 인자로 받은 vectorstore를 그대로 씁니다.
    k 값은 직접 정하고, 왜 그 값인지 주석으로 남기세요.
    """
    # TODO: vectorstore.as_retriever(...) 를 반환하세요.

    return vectorstore.as_retriever(search_kwargs={"k": 3})

    raise NotImplementedError("build_retriever 를 구현하세요.")


def build_rag_chain(retriever, llm=None):
    """질문 문자열을 받아 {"answer": str, "sources": list[str]} 를 반환하는 체인을 만듭니다.

    반드시 지킬 것:
      - 검색된 문서 본문을 프롬프트에 실제로 넣어야 합니다.
      - sources는 실제 검색 결과의 metadata에서 뽑습니다. 지어내면 안 됩니다.

    심화 파트를 구현했다면 assess_retrieval 판정을 거쳐야 합니다.
    """
    llm = llm or _default_llm()

    # TODO: 구현하세요.
    #   retriever.invoke(question) -> format_docs -> prompt -> llm -> 파싱
    #   반환은 반드시 {"answer": ..., "sources": [...]} 형태의 dict 입니다.
    #   RunnableLambda로 함수 하나를 감싸는 방식이 가장 간단합니다.
    from langchain_core.runnables import RunnableLambda
    system_rule = (
        "당신은 사내 규정 안내 도우미입니다. 아래 [문서]에 적힌 내용만 근거로 답하세요. "
        "문서에 없는 내용이면 '제공된 문서에서 답을 찾을 수 없습니다.'라고 답하세요."
    )
    def _answer(question) -> dict:
        """질문 하나를 받아 {"answer": str, "sources": list[str]} dict를 반환합니다.

        RunnableLambda가 이 함수를 감싸므로, chain.invoke(질문) 을 호출하면
        이 함수가 실행되고 반환한 dict가 그대로 결과가 됩니다.
        """
        
        question = "" if question is None else str(question)

        
        docs = list(retriever.invoke(question) or [])

       
        verdict = assess_retrieval(docs, question)
        if not verdict["usable"]:
            return {"answer": NO_ANSWER["answer"], "sources": []}

        
        context = format_docs(docs)
        messages = [
            ("system", system_rule + "\n\n[문서]\n" + context),
            ("human", question),
        ]

       
        answer = get_text(llm.invoke(messages))

       
        sources = list(dict.fromkeys(
            d.metadata.get("source") for d in docs if d.metadata.get("source")
        ))

        
        return {"answer": answer, "sources": sources}

    
    return RunnableLambda(_answer)

    raise NotImplementedError("build_rag_chain 을 구현하세요.")


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

def assess_retrieval(docs, question: str) -> dict:
    """검색 결과가 이 질문에 답하기에 쓸만한지 판정합니다.

    이 함수는 LLM을 호출하지 않습니다. 같은 입력에 항상 같은 출력을 내야 합니다.

    판정 방법은 자유입니다. 예를 들면
      - 질문에서 핵심어를 뽑아 문서 본문에 몇 개나 등장하는지 센다
      - 검색된 문서 개수와 길이를 본다
      - 특정 임계값을 넘는지 확인한다

    Returns:
        {"usable": bool, "reason": str, "matched": int} 형태의 dict.
        usable이 False면 build_rag_chain은 모델을 부르지 않고 거절해야 합니다.
    """
    # TODO: 구현하세요.

    if not docs:
        return {"usable": False, "reason": "검색 결과가 없습니다.", "matched": 0}

    
    stopwords = {"어떻", "무엇", "되나", "있나", "언제", "얼마", "알려", "궁금"}

    
    cleaned = "".join(c if c.isalnum() else " " for c in str(question or ""))

    
    keywords = []
    for tok in cleaned.split():
        if len(tok) < 2:
            continue
        key = tok[:2]
        if key in stopwords or key in keywords:
            continue
        keywords.append(key)

    if not keywords:
        return {"usable": False, "reason": "질문에서 핵심어를 찾지 못했습니다.", "matched": 0}

    
    body = " ".join(getattr(d, "page_content", "") for d in docs)
    matched = sum(1 for k in keywords if k in body)

    usable = matched >= 1
    reason = (
        f"핵심어 {len(keywords)}개 중 {matched}개가 문서에 등장합니다."
        if usable else "질문의 핵심어가 검색된 문서에 없습니다."
    )
    return {"usable": usable, "reason": reason, "matched": matched}

    raise NotImplementedError("assess_retrieval 을 구현하세요.")


NO_ANSWER = {
    "answer": "제공된 문서에서 답을 찾을 수 없습니다.",
    "sources": [],
}
