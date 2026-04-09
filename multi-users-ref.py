"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇 (Supabase Auth + Streamlit)

실행 (프로젝트 루트에서):
  poetry run streamlit run 7.MultiService/code/multi-users-ref.py

Streamlit Cloud → App Settings → Secrets:
  SUPABASE_URL
  SUPABASE_ANON_KEY  (멀티유저 RLS에는 anon 키 + 로그인 권장. 서비스 롤은 모든 행 접근 가능)

DB: 7.MultiService/code/multi-users-ref.sql 을 Supabase SQL Editor에서 실행

API 키(OpenAI / Anthropic / Gemini)는 사이드바에서 입력합니다(.env 미사용).
Supabase 로그인은 이메일+비밀번호(Supabase Auth)입니다. 로그인 ID 칸에 이메일을 입력하세요.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Generator, Iterable, List

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

MODEL_OPENAI = "gpt-4o-mini"
MODEL_CLAUDE = "claude-sonnet-4-5"
MODEL_GEMINI = "gemini-3-pro-preview"

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parents[1]


# ---------------------------------------------------------------------------
# Supabase 클라이언트 / 인증
# ---------------------------------------------------------------------------


def get_supabase_anon_key() -> str:
    return (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()


def _load_dotenv_like_file(path: Path, keys: set[str]) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k not in keys or os.getenv(k):
            continue
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ[k] = v


def load_local_supabase_env_fallback() -> None:
    needed = {"SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"}
    if os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
        return
    _load_dotenv_like_file(ROOT_DIR / ".env", needed)
    _load_dotenv_like_file(CODE_DIR / ".env", needed)


def create_supabase_client() -> Client:
    load_local_supabase_env_fallback()
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = get_supabase_anon_key()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL 과 SUPABASE_ANON_KEY(또는 SUPABASE_SERVICE_ROLE_KEY)가 "
            "환경 변수(Streamlit Secrets)에 필요합니다."
        )
    return create_client(url, key)


def restore_supabase_session(sb: Client) -> None:
    at = st.session_state.get("supabase_access_token")
    rt = st.session_state.get("supabase_refresh_token")
    if at and rt:
        try:
            sb.auth.set_session(at, rt)
        except Exception:
            st.session_state.pop("supabase_access_token", None)
            st.session_state.pop("supabase_refresh_token", None)


def persist_auth_session(sb: Client) -> None:
    sess = sb.auth.get_session()
    if sess and getattr(sess, "access_token", None) and getattr(sess, "refresh_token", None):
        st.session_state.supabase_access_token = sess.access_token
        st.session_state.supabase_refresh_token = sess.refresh_token


def clear_auth_session() -> None:
    st.session_state.pop("supabase_access_token", None)
    st.session_state.pop("supabase_refresh_token", None)


def current_user_id(sb: Client) -> str | None:
    sess = sb.auth.get_session()
    if not sess or not getattr(sess, "user", None):
        return None
    uid = getattr(sess.user, "id", None)
    return str(uid) if uid else None


def embedding_to_pgvector(emb: list[float]) -> str:
    return "[" + ",".join(str(x) for x in emb) + "]"


def clean_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


# ---------------------------------------------------------------------------
# LLM (스트리밍)
# ---------------------------------------------------------------------------


def openai_api_key() -> str:
    return (st.session_state.get("api_openai") or "").strip()


def anthropic_api_key() -> str:
    return (st.session_state.get("api_anthropic") or "").strip()


def google_api_key() -> str:
    return (st.session_state.get("api_google") or "").strip()


def apply_api_keys_to_environ() -> None:
    o, a, g = openai_api_key(), anthropic_api_key(), google_api_key()
    if o:
        os.environ["OPENAI_API_KEY"] = o
    if a:
        os.environ["ANTHROPIC_API_KEY"] = a
    if g:
        os.environ["GOOGLE_API_KEY"] = g


def build_title_llm_openai() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_OPENAI,
        temperature=0.2,
        streaming=False,
        api_key=openai_api_key() or None,
    )


def build_chat_llm(model_name: str, temperature: float = 0.2):
    if model_name == MODEL_OPENAI:
        return ChatOpenAI(
            model=MODEL_OPENAI,
            temperature=temperature,
            streaming=True,
            api_key=openai_api_key() or None,
        )
    if model_name == MODEL_CLAUDE:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=MODEL_CLAUDE,
            temperature=temperature,
            streaming=True,
            api_key=anthropic_api_key() or None,
        )
    if model_name == MODEL_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=MODEL_GEMINI,
            temperature=temperature,
            google_api_key=google_api_key() or None,
        )
    raise ValueError(f"지원하지 않는 모델: {model_name}")


def stream_llm_text(llm, messages: list[BaseMessage]) -> Generator[str, None, None]:
    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", None) or ""
        if isinstance(text, list):
            parts = []
            for p in text:
                if isinstance(p, dict) and "text" in p:
                    parts.append(p["text"])
                elif isinstance(p, str):
                    parts.append(p)
            text = "".join(parts)
        if text:
            yield text


# ---------------------------------------------------------------------------
# Supabase — 세션 / 메시지 (user_id 포함)
# ---------------------------------------------------------------------------


def db_upsert_session(sb: Client, session_id: str, title: str, user_id: str) -> None:
    sb.table("sessions").upsert(
        {"id": session_id, "title": title, "user_id": user_id},
        on_conflict="id",
    ).execute()


def db_list_sessions(sb: Client) -> list[dict[str, Any]]:
    r = (
        sb.table("sessions")
        .select("id,title,created_at,updated_at,user_id")
        .order("updated_at", desc=True)
        .execute()
    )
    return r.data or []


def db_delete_session(sb: Client, session_id: str) -> None:
    sb.table("sessions").delete().eq("id", session_id).execute()


def db_replace_messages(sb: Client, session_id: str, messages: list[dict[str, str]]) -> None:
    sb.table("messages").delete().eq("session_id", session_id).execute()
    rows = []
    for i, m in enumerate(messages):
        rows.append(
            {
                "session_id": session_id,
                "role": m["role"],
                "content": m["content"],
                "seq": i,
            }
        )
    if rows:
        sb.table("messages").insert(rows).execute()


def db_load_messages(sb: Client, session_id: str) -> list[dict[str, str]]:
    r = (
        sb.table("messages")
        .select("role,content,seq")
        .eq("session_id", session_id)
        .order("seq")
        .execute()
    )
    rows = r.data or []
    return [{"role": x["role"], "content": x["content"]} for x in rows]


def copy_session_snapshot(sb: Client, source_id: str, new_title: str, user_id: str) -> str:
    new_id = str(uuid.uuid4())
    sb.table("sessions").insert({"id": new_id, "title": new_title, "user_id": user_id}).execute()
    msgs = db_load_messages(sb, source_id)
    if msgs:
        db_replace_messages(sb, new_id, msgs)
    r = (
        sb.table("vector_documents")
        .select("content,metadata,embedding,file_name")
        .eq("session_id", source_id)
        .execute()
    )
    chunks = r.data or []
    batch: list[dict[str, Any]] = []
    for row in chunks:
        emb = row["embedding"]
        if isinstance(emb, list):
            emb = embedding_to_pgvector([float(x) for x in emb])
        batch.append(
            {
                "session_id": new_id,
                "content": row["content"],
                "metadata": row.get("metadata") or {},
                "embedding": emb,
                "file_name": row["file_name"],
            }
        )
    for i in range(0, len(batch), 50):
        sb.table("vector_documents").insert(batch[i : i + 50]).execute()
    return new_id


def list_vector_filenames(sb: Client, session_id: str) -> list[str]:
    r = (
        sb.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = sorted({row["file_name"] for row in (r.data or []) if row.get("file_name")})
    return names


# ---------------------------------------------------------------------------
# 벡터 저장 / 검색
# ---------------------------------------------------------------------------


def insert_documents_for_session(
    sb: Client,
    session_id: str,
    documents: Iterable[Document],
    embeddings: OpenAIEmbeddings,
    batch_size: int = 10,
) -> None:
    docs = list(documents)
    if not docs:
        return
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)
    rows: list[dict[str, Any]] = []
    for doc, vec in zip(docs, vectors, strict=True):
        fname = (
            doc.metadata.get("file_name")
            or doc.metadata.get("source")
            or "unknown.pdf"
        )
        if isinstance(fname, Path):
            fname = fname.name
        fname = str(fname) if fname else "unknown.pdf"
        rows.append(
            {
                "session_id": session_id,
                "content": doc.page_content,
                "metadata": clean_metadata(doc.metadata),
                "embedding": embedding_to_pgvector(vec),
                "file_name": fname,
            }
        )
    for i in range(0, len(rows), batch_size):
        sb.table("vector_documents").insert(rows[i : i + batch_size]).execute()


def retrieve_context(
    sb: Client,
    session_id: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    k: int = 5,
) -> tuple[str, list[Document]]:
    qvec = embeddings.embed_query(query)
    try:
        r = sb.rpc(
            "match_vector_documents",
            {
                "query_embedding": embedding_to_pgvector(qvec),
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        rows = r.data or []
    except Exception:
        r = (
            sb.table("vector_documents")
            .select("id,content,metadata,file_name")
            .eq("session_id", session_id)
            .limit(200)
            .execute()
        )
        rows = r.data or []

    docs: list[Document] = []
    for row in rows:
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        meta = {**meta, "file_name": row.get("file_name", "")}
        docs.append(Document(page_content=row.get("content", ""), metadata=meta))
    if not rows or "similarity" not in (rows[0] if rows else {}):
        docs = docs[:k]
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    return context, docs


# ---------------------------------------------------------------------------
# PDF / 제목
# ---------------------------------------------------------------------------


def process_pdf_uploads(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    session_id: str,
) -> list[Document]:
    all_docs: list[Document] = []
    tmp_dir = CODE_DIR / "_tmp_pdf"
    tmp_dir.mkdir(exist_ok=True)
    for uf in uploaded_files:
        safe_name = Path(uf.name).name
        path = tmp_dir / f"{session_id[:8]}_{safe_name}"
        path.write_bytes(uf.getbuffer())
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for d in pages:
            d.metadata = dict(d.metadata)
            d.metadata["file_name"] = safe_name
            d.metadata["session_id"] = session_id
        all_docs.extend(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)


def generate_session_title(llm_no_stream, first_q: str, first_a: str) -> str:
    sys = SystemMessage(
        content=(
            "다음은 챗봇의 첫 사용자 질문과 첫 답변입니다. "
            "이 대화를 한 줄로 요약하는 짧은 한국어 세션 제목(20자 이내)만 출력하세요. "
            "따옴표나 부가 설명 없이 제목만 쓰세요."
        )
    )
    human = HumanMessage(content=f"질문:\n{first_q}\n\n답변:\n{first_a[:2000]}")
    out = llm_no_stream.invoke([sys, human])
    text = (out.content or "").strip().split("\n")[0].strip()
    return text[:80] if text else "새 세션"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .msr-banner {
            background: linear-gradient(120deg, #1e3a5f 0%, #2563eb 45%, #7c3aed 100%);
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            color: #f8fafc;
            box-shadow: 0 4px 24px rgba(37, 99, 235, 0.25);
        }
        .msr-banner h1 {
            margin: 0;
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .msr-banner p {
            margin: 0.35rem 0 0 0;
            opacity: 0.92;
            font-size: 0.95rem;
        }
        .msr-logo {
            font-size: 2.2rem;
            line-height: 1;
        }
        div[data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        div[data-testid="stSidebarContent"] * {
            color: #e2e8f0 !important;
        }
        div[data-testid="stSidebarContent"] .stSelectbox label,
        div[data-testid="stSidebarContent"] .stMultiSelect label {
            color: #cbd5e1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.12, 0.88])
    with c1:
        st.markdown('<p class="msr-logo">💬</p>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            """
            <div class="msr-banner">
              <h1>PDF 기반 멀티유저 멀티세션 RAG 챗봇</h1>
              <p>Supabase에 계정·세션·벡터를 저장하고, 여러 LLM 중에서 선택해 문서 기반으로 답합니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def ensure_session_row(sb: Client, session_id: str, user_id: str, title: str = "새 세션") -> None:
    db_upsert_session(sb, session_id, title, user_id)


def persist_full_state(sb: Client, user_id: str) -> None:
    sid = st.session_state.session_id
    ensure_session_row(sb, sid, user_id, st.session_state.get("session_title") or "새 세션")
    db_replace_messages(sb, sid, st.session_state.messages)


def maybe_update_title(sb: Client, llm_title, user_id: str) -> None:
    msgs = st.session_state.messages
    if len(msgs) < 2 or st.session_state.get("title_locked"):
        return
    first_u = next((m["content"] for m in msgs if m["role"] == "user"), None)
    first_a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if not first_u or not first_a:
        return
    title = generate_session_title(llm_title, first_u, first_a)
    st.session_state.session_title = title
    st.session_state.title_locked = True
    db_upsert_session(sb, st.session_state.session_id, title, user_id)


def load_session_ui(sb: Client, session_id: str) -> None:
    st.session_state.session_id = session_id
    st.session_state.messages = db_load_messages(sb, session_id)
    st.session_state.title_locked = True
    sess = next((s for s in db_list_sessions(sb) if s["id"] == session_id), None)
    st.session_state.session_title = (sess or {}).get("title") or "세션"


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_title" not in st.session_state:
        st.session_state.session_title = "새 세션"
    if "title_locked" not in st.session_state:
        st.session_state.title_locked = False
    if "model_name" not in st.session_state:
        st.session_state.model_name = MODEL_OPENAI
    if "last_multi" not in st.session_state:
        st.session_state.last_multi = []
    for k in ("api_openai", "api_anthropic", "api_google"):
        if k not in st.session_state:
            st.session_state[k] = ""


def label_for_session(row: dict[str, Any]) -> str:
    return f"{row['title']} ({str(row['id'])[:8]})"


def render_api_key_inputs() -> None:
    st.markdown("### API 키")
    st.caption("OpenAI는 임베딩에 필수입니다. 선택한 LLM에 맞게 입력하세요.")
    st.text_input(
        "OpenAI API Key",
        type="password",
        key="api_openai",
    )
    st.text_input(
        "Anthropic API Key",
        type="password",
        key="api_anthropic",
    )
    st.text_input(
        "Google (Gemini) API Key",
        type="password",
        key="api_google",
    )


def render_auth_sidebar(sb: Client) -> bool:
    st.markdown("### 로그인 / 회원가입")
    st.caption("로그인 ID에는 Supabase에 등록된 이메일을 입력하세요.")
    login_id = st.text_input("로그인 ID (이메일)", key="auth_login_id")
    password = st.text_input("비밀번호", type="password", key="auth_password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("로그인", use_container_width=True):
            if not login_id or not password:
                st.warning("이메일과 비밀번호를 입력하세요.")
            else:
                try:
                    r = sb.auth.sign_in_with_password(
                        {"email": login_id.strip(), "password": password}
                    )
                    sess = getattr(r, "session", None)
                    if sess and getattr(sess, "access_token", None):
                        st.session_state.supabase_access_token = sess.access_token
                        rt = getattr(sess, "refresh_token", None)
                        if rt:
                            st.session_state.supabase_refresh_token = rt
                    persist_auth_session(sb)
                    st.success("로그인했습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"로그인 실패: {e}")
    with c2:
        if st.button("회원가입", use_container_width=True):
            if not login_id or not password:
                st.warning("이메일과 비밀번호를 입력하세요.")
            else:
                try:
                    sb.auth.sign_up({"email": login_id.strip(), "password": password})
                    st.info(
                        "가입 요청을 보냈습니다. Supabase에서 이메일 확인을 켠 경우 "
                        "메일함을 확인한 뒤 로그인하세요."
                    )
                except Exception as e:
                    st.error(f"회원가입 실패: {e}")

    if st.button("로그아웃", use_container_width=True):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        clear_auth_session()
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.session_title = "새 세션"
        st.session_state.title_locked = False
        st.rerun()

    uid = current_user_id(sb)
    return uid is not None


def main() -> None:
    st.set_page_config(
        page_title="PDF 멀티유저 RAG",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    inject_styles()

    try:
        sb = create_supabase_client()
        restore_supabase_session(sb)
    except Exception as e:
        st.error(str(e))
        return

    with st.sidebar:
        render_api_key_inputs()
        apply_api_keys_to_environ()
        render_auth_sidebar(sb)
        user_id = current_user_id(sb)

    if not openai_api_key():
        st.warning("사이드바에서 OpenAI API 키를 입력하세요. (텍스트 임베딩에 사용됩니다)")
        if not user_id:
            st.info("로그인 후 채팅과 세션 저장을 사용할 수 있습니다.")
        return

    if not user_id:
        st.info("로그인 후 채팅과 세션 저장을 사용할 수 있습니다.")
        return

    embeddings = OpenAIEmbeddings(api_key=openai_api_key())

    if st.session_state.model_name == MODEL_CLAUDE and not anthropic_api_key():
        st.warning("Claude 사용 시 사이드바에 Anthropic API 키를 입력하세요.")
    if st.session_state.model_name == MODEL_GEMINI and not google_api_key():
        st.warning("Gemini 사용 시 사이드바에 Google API 키를 입력하세요.")

    with st.sidebar:
        st.divider()
        st.markdown("### 모델 선택")
        st.session_state.model_name = st.radio(
            "LLM",
            options=[MODEL_OPENAI, MODEL_CLAUDE, MODEL_GEMINI],
            index=[MODEL_OPENAI, MODEL_CLAUDE, MODEL_GEMINI].index(st.session_state.model_name)
            if st.session_state.model_name in (MODEL_OPENAI, MODEL_CLAUDE, MODEL_GEMINI)
            else 0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### 세션 관리")
        sessions = db_list_sessions(sb)
        labels_map: dict[str, str] = {label_for_session(s): s["id"] for s in sessions}
        label_list = list(labels_map.keys())

        def on_selectbox_change() -> None:
            lab = st.session_state.get("sess_select_label")
            if lab and lab in labels_map:
                load_session_ui(sb, labels_map[lab])

        st.selectbox(
            "세션 (선택 시 내용 표시)",
            options=[""] + label_list,
            format_func=lambda x: "— 선택 —" if x == "" else x,
            key="sess_select_label",
            on_change=on_selectbox_change,
        )

        multi_labels = st.multiselect(
            "세션 멀티 선택 (선택 시 자동 로드)",
            options=label_list,
            default=[],
            key="sess_multi",
        )
        if multi_labels != st.session_state.last_multi:
            st.session_state.last_multi = list(multi_labels)
            if multi_labels:
                sid = labels_map[multi_labels[-1]]
                load_session_ui(sb, sid)
                row = next(s for s in sessions if s["id"] == sid)
                st.session_state.sess_select_label = label_for_session(row)
                st.rerun()

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("세션저장", use_container_width=True):
                msgs = st.session_state.messages
                if len(msgs) < 2:
                    st.warning("저장할 대화(질문·답변)가 없습니다.")
                else:
                    persist_full_state(sb, user_id)
                    first_u = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    first_a = next(
                        (m["content"] for m in msgs if m["role"] == "assistant"), ""
                    )
                    title = generate_session_title(
                        build_title_llm_openai(), first_u, first_a
                    )
                    new_id = copy_session_snapshot(sb, st.session_state.session_id, title, user_id)
                    st.success(f"새 세션으로 저장했습니다. id: {new_id[:8]}…")
                    st.rerun()
        with col_b:
            if st.button("세션로드", use_container_width=True):
                lab = st.session_state.get("sess_select_label") or (
                    multi_labels[-1] if multi_labels else None
                )
                if not lab or lab not in labels_map:
                    st.warning("로드할 세션을 먼저 선택하세요.")
                else:
                    load_session_ui(sb, labels_map[lab])
                    st.success("세션을 불러왔습니다.")
                    st.rerun()

        if st.button("세션삭제", use_container_width=True):
            lab = st.session_state.get("sess_select_label")
            if not lab or lab not in labels_map:
                st.warning("삭제할 세션을 선택하세요.")
            else:
                db_delete_session(sb, labels_map[lab])
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.session_title = "새 세션"
                st.session_state.title_locked = False
                st.success("삭제했습니다.")
                st.rerun()

        if st.button("화면초기화", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.session_title = "새 세션"
            st.session_state.title_locked = False
            st.success("화면을 초기화했습니다. (DB의 기존 세션은 그대로입니다)")
            st.rerun()

        if st.button("vectordb", use_container_width=True):
            names = list_vector_filenames(sb, st.session_state.session_id)
            if not names:
                st.info("현재 세션에 저장된 파일명이 없습니다.")
            else:
                st.write("**현재 세션 벡터 DB 파일명**")
                for n in names:
                    st.write(f"- {n}")

        st.divider()
        st.markdown("### PDF 업로드")
        files = st.file_uploader(
            "PDF (여러 개)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("문서 임베딩", use_container_width=True):
            if not files:
                st.warning("PDF를 선택하세요.")
            else:
                with st.spinner("PDF 처리 및 Supabase 저장 중…"):
                    ensure_session_row(
                        sb,
                        st.session_state.session_id,
                        user_id,
                        st.session_state.session_title,
                    )
                    docs = process_pdf_uploads(list(files), st.session_state.session_id)
                    insert_documents_for_session(
                        sb, st.session_state.session_id, docs, embeddings
                    )
                    persist_full_state(sb, user_id)
                st.success("임베딩을 저장했습니다. (자동 세션 저장 완료)")
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("문서에 대해 질문해 주세요.")
    if not user_q:
        return

    try:
        llm = build_chat_llm(st.session_state.model_name)
    except Exception as e:
        st.error(str(e))
        return

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    context, _src_docs = retrieve_context(
        sb, st.session_state.session_id, user_q, embeddings
    )
    system_text = (
        "너는 업로드된 문서를 우선 근거로 답하는 한국어 어시스턴트다. "
        "근거가 부족하면 추측하지 말고 말한다. "
        "답변은 친절한 존댓말로 작성한다.\n\n"
        "반드시 답변 마지막에 다음 형식을 지킨다:\n\n"
        "---\n"
        "**추가로 물어볼 만한 질문**\n"
        "1. …\n"
        "2. …\n"
        "3. …\n\n"
        "위 세 가지 질문은 사용자가 다음에 탐구하면 좋은 내용이어야 한다.\n\n"
        f"### 문서 발췌\n{context if context else '(저장된 청크 없음)'}"
    )

    messages_lc: list[BaseMessage] = [SystemMessage(content=system_text)]
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            messages_lc.append(HumanMessage(content=m["content"]))
        else:
            messages_lc.append(AIMessage(content=m["content"]))
    messages_lc.append(HumanMessage(content=user_q))

    with st.chat_message("assistant"):
        buf: list[str] = []

        def token_gen() -> Generator[str, None, None]:
            for t in stream_llm_text(llm, messages_lc):
                buf.append(t)
                yield t

        try:
            st.write_stream(token_gen())
        except Exception:
            buf.clear()
            acc = ""
            for t in stream_llm_text(llm, messages_lc):
                acc += t
            st.markdown(acc)
            buf.append(acc)
        answer = "".join(buf)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    ensure_session_row(
        sb,
        st.session_state.session_id,
        user_id,
        st.session_state.session_title,
    )
    persist_full_state(sb, user_id)
    maybe_update_title(sb, build_title_llm_openai(), user_id)


if __name__ == "__main__":
    main()
