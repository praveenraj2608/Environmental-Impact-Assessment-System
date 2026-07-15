"""
Page 9 — AI Environmental Consultant
RAG-powered chat interface with full debug panel and per-stage timing.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="AI Consultant — EIA",
    page_icon="🤖",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent.parent / "assets" / "css" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
/* ─── Chat bubbles ─────────────────────────────────────────────────────────── */
.chat-user {
    display: flex; justify-content: flex-end; margin: 0.6rem 0;
}
.chat-user .bubble {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #f0f9ff; padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px; max-width: 72%;
    font-size: 0.92rem; line-height: 1.55;
    box-shadow: 0 2px 12px rgba(37,99,235,0.35);
}
.chat-ai { display: flex; justify-content: flex-start; margin: 0.6rem 0; }
.chat-ai .bubble {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(6,182,212,0.25); color: #e2e8f0;
    padding: 0.85rem 1.15rem;
    border-radius: 18px 18px 18px 4px; max-width: 80%;
    font-size: 0.92rem; line-height: 1.65;
    box-shadow: 0 2px 14px rgba(6,182,212,0.12);
}
.chat-ai .avatar {
    font-size: 1.6rem; margin-right: 0.55rem;
    flex-shrink: 0; align-self: flex-end;
}
/* ─── Confidence badge ──────────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 0.18rem 0.65rem;
    border-radius: 99px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; margin-left: 0.5rem;
}
.badge-high   { background:rgba(16,185,129,0.2); color:#10b981; border:1px solid #10b98160; }
.badge-medium { background:rgba(245,158,11,0.2);  color:#f59e0b; border:1px solid #f59e0b60; }
.badge-low    { background:rgba(239,68,68,0.2);   color:#ef4444; border:1px solid #ef444460; }
/* ─── Status pills ─────────────────────────────────────────────────────────── */
.rag-status-row { display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem; }
.rag-pill {
    display:flex; align-items:center; gap:0.4rem;
    padding:0.35rem 0.9rem; border-radius:99px;
    font-size:0.8rem; font-weight:600; border:1px solid;
}
.pill-green  { background:rgba(16,185,129,0.12); color:#10b981; border-color:#10b98140; }
.pill-red    { background:rgba(239,68,68,0.12);  color:#ef4444; border-color:#ef444440; }
.pill-yellow { background:rgba(245,158,11,0.12); color:#f59e0b; border-color:#f59e0b40; }
/* ─── Source card ──────────────────────────────────────────────────────────── */
.src-card {
    background:rgba(15,23,42,0.7); border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:0.65rem 0.9rem; margin:0.35rem 0;
}
.src-title   { color:#93c5fd; font-weight:600; font-size:0.85rem; }
.src-meta    { color:#475569; font-size:0.76rem; margin-top:0.15rem; }
.src-excerpt { color:#94a3b8; font-size:0.8rem; margin-top:0.35rem; font-style:italic; }
/* ─── Debug panel ───────────────────────────────────────────────────────────── */
.dbg-panel {
    background:rgba(2,6,23,0.9); border:1px solid rgba(99,102,241,0.3);
    border-radius:12px; padding:0.9rem 1.1rem; font-family:'Fira Code',monospace;
    font-size:0.78rem;
}
.dbg-row { display:flex; justify-content:space-between; padding:0.18rem 0;
           border-bottom:1px solid rgba(255,255,255,0.04); }
.dbg-label { color:#64748b; }
.dbg-val   { color:#22d3ee; font-weight:600; }
.dbg-val-warn { color:#f59e0b; font-weight:600; }
.dbg-val-ok   { color:#10b981; font-weight:600; }
/* ─── Timing bar ─────────────────────────────────────────────────────────────── */
.timing-bar {
    display:flex; gap:0.3rem; align-items:center;
    font-size:0.72rem; color:#475569; margin-top:0.4rem;
    padding-top:0.35rem; border-top:1px solid rgba(255,255,255,0.06);
}
.t-seg {
    padding:0.12rem 0.45rem; border-radius:4px;
    background:rgba(99,102,241,0.15); color:#a5b4fc; font-weight:600;
}
.t-seg-slow { background:rgba(239,68,68,0.15); color:#fca5a5; }
/* ─── Page header ─────────────────────────────────────────────────────────── */
.ai-header {
    background:linear-gradient(135deg,
        rgba(6,182,212,0.12) 0%, rgba(99,102,241,0.12) 50%,
        rgba(16,185,129,0.08) 100%);
    border:1px solid rgba(6,182,212,0.2); border-radius:18px;
    padding:2rem 2.5rem 1.5rem; margin-bottom:1.5rem;
    position:relative; overflow:hidden;
}
.ai-header::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:180px; height:180px; border-radius:50%;
    background:radial-gradient(circle, rgba(6,182,212,0.15), transparent 70%);
}
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ai-header">
  <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
    <span style="font-size:2.8rem; filter:drop-shadow(0 0 20px rgba(6,182,212,0.6));">🤖</span>
    <div>
      <h1 style="font-family:'Outfit',sans-serif; font-weight:800;
                 background:linear-gradient(135deg,#93c5fd,#22d3ee,#a5f3fc);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 background-clip:text; font-size:2rem; margin:0; letter-spacing:-0.02em;">
        AI Environmental Consultant
      </h1>
      <p style="color:#64748b; font-size:0.9rem; margin:0.2rem 0 0;">
        RAG-powered · Knowledge-grounded · Per-stage timing
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── RAG pipeline — cached once per session ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    from rag.rag_pipeline import RAGPipeline
    return RAGPipeline()


# ── Session state init ─────────────────────────────────────────────────────────
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized   = False
    st.session_state.rag_init_status   = {}
    st.session_state.rag_chat_history  = []
    st.session_state.rag_pending_input = ""

pipeline = get_rag_pipeline()

# ── First-run index build ──────────────────────────────────────────────────────
if not st.session_state.rag_initialized:
    init_box = st.empty()
    with init_box.container():
        st.markdown("""
        <div style='text-align:center; padding:2rem;'>
            <div style='font-size:3rem; display:inline-block;'>📚</div>
            <div style='color:#22d3ee; font-weight:700; font-size:1.1rem; margin-top:0.5rem;'>
                Building Knowledge Index…
            </div>
            <div style='color:#475569; font-size:0.85rem; margin-top:0.3rem;'>
                First run embeds all documents — subsequent startups load instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)
        prog_bar = st.progress(0.0, text="Initializing …")

        def _progress(msg: str, frac: float):
            prog_bar.progress(min(frac, 1.0), text=msg)

        pipeline.knowledge_manager.progress_callback = _progress
        status = pipeline.initialize()
        st.session_state.rag_init_status = status
        st.session_state.rag_initialized = True

    init_box.empty()
    if st.session_state.rag_init_status.get("status") == "error":
        st.error(f"⚠️ KB error: {st.session_state.rag_init_status.get('message')}")


# ── Status bar ─────────────────────────────────────────────────────────────────
ollama_st   = pipeline.ollama_status()
kb_st       = pipeline.knowledge_base_status()
chunk_count = kb_st.get("chunk_count", 0)

olm_pill = (
    f'<div class="rag-pill pill-green">🟢 Ollama: {ollama_st["current"]} ready</div>'
    if ollama_st["available"] else
    '<div class="rag-pill pill-red">🔴 Ollama offline — run `ollama serve`</div>'
)
kb_pill = (
    f'<div class="rag-pill pill-green">🟢 KB: {chunk_count:,} chunks indexed</div>'
    if kb_st["initialized"] and chunk_count > 0 else
    '<div class="rag-pill pill-yellow">🟡 KB: indexed but empty</div>'
    if kb_st["initialized"] else
    '<div class="rag-pill pill-red">🔴 KB not indexed</div>'
)
st.markdown(f'<div class="rag-status-row">{olm_pill}{kb_pill}</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Quick-Start Questions")
    quick_questions = [
        ("🫁", "What is PM2.5 and why is it harmful?"),
        ("📏", "What are WHO air quality standards?"),
        ("🏭", "Suggest mitigation for industrial pollution"),
        ("☁️", "How does climate change affect air quality?"),
        ("⚖️", "What environmental regulations apply to industries?"),
        ("🔬", "What research links PM2.5 to cardiovascular disease?"),
        ("🌱", "Suggest nature-based solutions for air pollution"),
        ("📊", "Explain the AQI scale and health thresholds"),
    ]
    for icon, question in quick_questions:
        if st.button(f"{icon} {question}", key=f"qs_{question[:22]}",
                     use_container_width=True):
            st.session_state.rag_pending_input = question

    st.markdown("---")

    # Prediction context toggle
    has_prediction = "assessment" in st.session_state
    use_context = st.toggle(
        "🔗 Inject prediction context", value=has_prediction,
        help="Include current session predictions in the query.",
    )
    if has_prediction and use_context:
        a = st.session_state["assessment"]
        st.caption(
            f"Area: {a.get('city_type','—')} · "
            f"Health: {a.get('health_impact','—')} · "
            f"Pollution: {a.get('pollution_level','—')}"
        )
    elif not has_prediction:
        st.caption("Run Pages 3–6 first to load prediction context.")

    st.markdown("---")

    # Debug panel toggle
    show_debug = st.toggle("🔍 Show Debug Panel", value=False,
                           help="Display per-stage timing and token counts.")

    st.markdown("---")

    if st.button("🔄 Rebuild Knowledge Index", use_container_width=True):
        st.session_state.rag_initialized = False
        st.session_state.rag_init_status = {}
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**⚙️ RAG Config**")
    from rag.config import (
        OLLAMA_MODEL, EMBEDDING_MODEL, RETRIEVAL_TOP_K,
        SIMILARITY_THRESHOLD, CHUNK_SIZE, OLLAMA_MAX_TOKENS,
        OLLAMA_TEMPERATURE,
    )
    st.caption(
        f"LLM: `{OLLAMA_MODEL}` · max_tokens=`{OLLAMA_MAX_TOKENS}` · temp=`{OLLAMA_TEMPERATURE}`  \n"
        f"Embed: `{EMBEDDING_MODEL.split('/')[-1]}`  \n"
        f"Top-K: `{RETRIEVAL_TOP_K}` · Threshold: `{SIMILARITY_THRESHOLD}` · Chunk: `{CHUNK_SIZE}` chars"
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _timing_bar_html(timing: dict) -> str:
    """Build an inline timing bar for display inside a chat bubble."""
    if not timing:
        return ""
    embed  = timing.get("embed_secs",     0)
    ret    = timing.get("retrieval_secs", 0)
    llm    = timing.get("llm_secs",       0)
    total  = timing.get("total_secs",     0)

    slow_cls = "t-seg-slow" if llm > 8 else "t-seg"

    return (
        f'<div class="timing-bar">'
        f'⏱ <span class="t-seg">embed {embed:.2f}s</span>'
        f'<span class="t-seg">search {ret:.2f}s</span>'
        f'<span class="{slow_cls}">LLM {llm:.2f}s</span>'
        f'<span style="margin-left:auto; color:#334155;">total {total:.2f}s</span>'
        f'</div>'
    )


def _debug_panel(timing: dict) -> None:
    """Render the Step 11 debug panel as a styled card."""
    if not timing:
        st.caption("No timing data available.")
        return

    embed  = timing.get("embed_secs",       0)
    ret    = timing.get("retrieval_secs",   0)
    prompt = timing.get("prompt_secs",      0)
    llm    = timing.get("llm_secs",         0)
    total  = timing.get("total_secs",       0)
    ftok   = timing.get("first_token_secs")
    tokens = timing.get("est_prompt_tokens", 0)
    chars  = timing.get("context_chars",    0)
    chunks = timing.get("chunks_used",      0)

    def _cls(val, warn, bad):
        if val >= bad:   return "dbg-val-warn"
        if val >= warn:  return "dbg-val-warn"
        return "dbg-val-ok"

    rows = [
        ("Embedding time",        f"{embed:.3f}s",         _cls(embed,  0.2, 0.5)),
        ("Retrieval time",        f"{ret:.3f}s",           _cls(ret,    0.1, 0.3)),
        ("Prompt build time",     f"{prompt:.3f}s",        "dbg-val"),
        ("Prompt tokens (est.)",  f"~{tokens}",            _cls(tokens, 800, 1200)),
        ("Context chars",         f"{chars}",              _cls(chars,  2000, 2500)),
        ("Chunks used",           f"{chunks}",             "dbg-val"),
        ("LLM generation time",   f"{llm:.3f}s",           _cls(llm,    5.0, 10.0)),
        ("First token latency",   f"{ftok:.3f}s" if ftok else "—", "dbg-val"),
        ("Total response time",   f"{total:.3f}s",         _cls(total,  6.0, 12.0)),
    ]

    html_rows = "".join(
        f'<div class="dbg-row">'
        f'<span class="dbg-label">{label}</span>'
        f'<span class="{cls}">{val}</span>'
        f'</div>'
        for label, val, cls in rows
    )
    st.markdown(
        f'<div class="dbg-panel">{html_rows}</div>',
        unsafe_allow_html=True,
    )


# ── Main chat area ─────────────────────────────────────────────────────────────
chat_col, dbg_col = st.columns([3, 1]) if show_debug else (st.container(), None)

with chat_col:
    history_container = st.container()

    with history_container:
        if not st.session_state.rag_chat_history:
            st.markdown("""
            <div style='text-align:center; padding:2.5rem 1rem; color:#334155;'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>💬</div>
                <div style='font-size:1rem; font-weight:600; color:#475569;'>No messages yet</div>
                <div style='font-size:0.85rem; margin-top:0.3rem;'>
                    Use a quick-start button or type a question below.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for turn in st.session_state.rag_chat_history:
                # User bubble
                st.markdown(
                    f'<div class="chat-user"><div class="bubble">🧑 {turn["question"]}</div></div>',
                    unsafe_allow_html=True,
                )

                # AI bubble with inline timing bar
                confidence = turn.get("confidence", "Low")
                badge_cls  = f"badge-{confidence.lower()}"
                answer_html = turn["answer"].replace("\n", "<br>")
                timing_bar  = _timing_bar_html(turn.get("timing", {}))

                st.markdown(
                    f'<div class="chat-ai">'
                    f'<span class="avatar">🤖</span>'
                    f'<div class="bubble">'
                    f'{answer_html}'
                    f'<div style="margin-top:0.5rem; border-top:1px solid rgba(255,255,255,0.07); '
                    f'padding-top:0.35rem; font-size:0.75rem; color:#475569;">'
                    f'<span class="badge {badge_cls}">{confidence} confidence</span>'
                    f'&nbsp;·&nbsp;{turn.get("model","AI")}'
                    f'</div>'
                    f'{timing_bar}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Sources expander
                if turn.get("sources"):
                    with st.expander(f"📚 {len(turn['sources'])} source(s) cited", expanded=False):
                        for src in turn["sources"]:
                            sim_pct = int(src["similarity"] * 100)
                            sim_color = (
                                "#10b981" if src["similarity"] >= 0.65
                                else "#f59e0b" if src["similarity"] >= 0.40
                                else "#ef4444"
                            )
                            st.markdown(f"""
                            <div class="src-card">
                                <div class="src-title">📄 {src['filename']}</div>
                                <div class="src-meta">
                                    {src['category']}
                                    &nbsp;·&nbsp;
                                    <span style="color:{sim_color}; font-weight:700;">{sim_pct}% relevance</span>
                                </div>
                                <div class="src-excerpt">{src.get('excerpt','')}</div>
                            </div>
                            """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Input row ─────────────────────────────────────────────────────────────
    default_input = st.session_state.pop("rag_pending_input", "")

    inp_col, btn_col, clr_col = st.columns([6, 1, 1])
    with inp_col:
        user_input = st.text_input(
            label="Ask the AI Consultant…",
            value=default_input,
            placeholder="e.g. What is PM2.5?",
            label_visibility="collapsed",
            key="rag_text_input",
        )
    with btn_col:
        send = st.button("Send ➤", use_container_width=True, type="primary")
    with clr_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.rag_chat_history = []
            st.rerun()


# ── Debug panel — right column (Step 11) ───────────────────────────────────────
if show_debug and dbg_col is not None:
    with dbg_col:
        st.markdown("#### 🔍 Debug Panel")
        if st.session_state.rag_chat_history:
            last = st.session_state.rag_chat_history[-1]
            _debug_panel(last.get("timing", {}))

            # Benchmark targets
            st.markdown("**Targets**")
            targets = [
                ("Embed",    last.get("timing", {}).get("embed_secs",     0), 0.20),
                ("Retrieval",last.get("timing", {}).get("retrieval_secs", 0), 0.10),
                ("LLM",      last.get("timing", {}).get("llm_secs",       0), 5.00),
                ("Total",    last.get("timing", {}).get("total_secs",     0), 6.00),
            ]
            for label, val, target in targets:
                pct = min(val / target, 1.0) if target else 1.0
                color = "#10b981" if val <= target else "#ef4444"
                st.markdown(
                    f'<div style="font-size:0.78rem; color:#64748b; margin-top:0.3rem;">'
                    f'{label}: <span style="color:{color};">{val:.2f}s</span> / {target}s target</div>',
                    unsafe_allow_html=True,
                )
                st.progress(pct)
        else:
            st.caption("Ask a question to see debug data.")

        st.markdown("---")
        st.markdown("**Model info**")
        st.caption(
            f"LLM: `{OLLAMA_MODEL}`  \n"
            f"Embed: `{EMBEDDING_MODEL.split('/')[-1]}`  \n"
            f"Top-K: `{RETRIEVAL_TOP_K}`  \n"
            f"Max tokens: `{OLLAMA_MAX_TOKENS}`"
        )


# ── Handle send ────────────────────────────────────────────────────────────────
if (send or default_input) and user_input.strip():
    if not ollama_st["available"]:
        st.warning(
            f"⚠️ Ollama offline. Run `ollama serve` and `ollama pull {OLLAMA_MODEL}`."
        )
    else:
        pred_ctx = None
        if use_context and has_prediction:
            a = st.session_state["assessment"]
            pred_ctx = {
                "city_type":       a.get("city_type"),
                "health_impact":   a.get("health_impact"),
                "pollution_level": a.get("pollution_level"),
                "risk_level":      a.get("risk_level"),
                "trend":           a.get("trend"),
            }

        with st.spinner("🤖 Thinking …"):
            t_wall = time.perf_counter()
            result = pipeline.ask(
                question=user_input.strip(),
                prediction_context=pred_ctx,
            )
            wall = time.perf_counter() - t_wall

        st.session_state.rag_chat_history.append(
            {
                "question":   user_input.strip(),
                "answer":     result.get("answer", "No response."),
                "sources":    result.get("sources", []),
                "confidence": result.get("confidence", "Low"),
                "model":      result.get("model", OLLAMA_MODEL),
                "timing":     result.get("timing", {}),
            }
        )
        st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25);
            border-radius:12px; padding:0.85rem 1.25rem; font-size:0.82rem; color:#fca5a5;'>
    ⚠️ <strong>Disclaimer:</strong> AI responses are for informational purposes only.
    They do not constitute legal, medical, or regulatory advice.
</div>
""", unsafe_allow_html=True)
