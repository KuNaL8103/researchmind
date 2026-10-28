import json
import time

import streamlit as st

from src.pipeline import run_research_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 1rem;
    opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0ebe0;
    margin: 0 0 1rem;
}
.hero h1 span {
    color: #ff8c32;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #a09890;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,50,0.15);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}

/* ── Streamlit input overrides ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important;
    color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
.stTextInput > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #ff8c32 !important;
    font-weight: 500 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.2rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,140,50,0.4) !important;
    opacity: 0.95 !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Pipeline step cards ── */
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.step-card.active {
    border-color: rgba(255,140,50,0.4);
    background: rgba(255,140,50,0.04);
}
.step-card.done {
    border-color: rgba(80,200,120,0.3);
    background: rgba(80,200,120,0.03);
}
.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 14px 0 0 14px;
    background: rgba(255,255,255,0.05);
    transition: background 0.3s;
}
.step-card.active::before { background: #ff8c32; }
.step-card.done::before   { background: #50c878; }

.step-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.3rem;
}
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #ff8c32;
    opacity: 0.7;
}
.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f0ebe0;
}
.step-status {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}
.status-waiting  { color: #555; }
.status-running  { color: #ff8c32; }
.status-done     { color: #50c878; }

/* ── Result panels ── */
.result-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}
.result-panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,140,50,0.15);
}
.result-content {
    font-size: 0.92rem;
    line-height: 1.8;
    color: #cdc8bf;
    white-space: pre-wrap;
    font-family: 'DM Sans', sans-serif;
}

/* ── Report & feedback panels ── */
.report-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,140,50,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1rem;
}
.feedback-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(80,200,120,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1rem;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
}
.panel-label.orange {
    color: #ff8c32;
    border-bottom: 1px solid rgba(255,140,50,0.15);
}
.panel-label.green {
    color: #50c878;
    border-bottom: 1px solid rgba(80,200,120,0.15);
}

/* ── Progress text ── */
.stSpinner > div { color: #ff8c32 !important; }

/* ── Expander ── */
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #a09890 !important;
    letter-spacing: 0.1em !important;
    cursor: pointer;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0ebe0;
    margin: 2rem 0 1rem;
}

/* ── Metrics row ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.8rem;
    margin: 1rem 0 1.5rem;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,50,0.16);
    border-radius: 12px;
    padding: 1rem 1.1rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0ebe0;
    line-height: 1.1;
}
.metric-value.green { color: #50c878; }
.metric-value.orange { color: #ff8c32; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #a09890;
    margin-top: 0.45rem;
}
.metric-sub {
    font-size: 0.72rem;
    color: #706860;
    margin-top: 0.25rem;
}

/* ── Toast-style notice ── */
.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #605850;
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: render quantitative metrics ──────────────────────────────────────
def render_metrics(metrics: dict, evaluation: dict):
    latency = metrics.get("latency_seconds") or 0
    reduction = metrics.get("token_reduction_pct") or 0
    saved = metrics.get("tokens_saved") or 0
    chunks = metrics.get("chunks_indexed") or 0
    retrieved = metrics.get("retrieved_chunks") or 0
    retrieved_sources = metrics.get("retrieved_sources") or 0
    critic = evaluation.get("critic_score")
    grounded = evaluation.get("groundedness_score") or 0
    verdict = evaluation.get("verdict", "Not evaluated")
    sources = metrics.get("sources_scraped") or 0

    grounded_cls = "green" if grounded >= 0.6 else "orange"
    critic_text = f"{critic:g}/10" if critic is not None else "—"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{latency:.1f}s</div>
            <div class="metric-label">Pipeline Latency</div>
            <div class="metric-sub">end-to-end wall clock</div>
        </div>
        <div class="metric-card">
            <div class="metric-value green">{reduction:.1f}%</div>
            <div class="metric-label">Token Reduction</div>
            <div class="metric-sub">{saved:,} tokens saved by RAG</div>
        </div>
        <div class="metric-card">
            <div class="metric-value orange">{critic_text}</div>
            <div class="metric-label">Critic Score</div>
            <div class="metric-sub">1–10 publication review</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {grounded_cls}">{grounded * 100:.0f}%</div>
            <div class="metric-label">Groundedness</div>
            <div class="metric-sub">{verdict}</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{retrieved}/{chunks}</div>
            <div class="metric-label">Chunks Retrieved</div>
            <div class="metric-sub">{retrieved_sources} unique URLs · {sources} scraped</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Evaluation & telemetry details", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Stage latency (seconds)**")
            stages = metrics.get("stage_latency_seconds") or {}
            if stages:
                for stage, seconds in stages.items():
                    st.markdown(f"- `{stage}` — {seconds:.3f}s")
            else:
                st.caption("No stage timings recorded.")

            st.markdown("**RAG configuration**")
            st.markdown(
                f"- Embedding: `{metrics.get('embedding_provider') or 'n/a'}` / "
                f"`{metrics.get('embedding_model') or 'n/a'}`\n"
                f"- Chunk size: `{metrics.get('chunk_size')}` · overlap: "
                f"`{metrics.get('chunk_overlap')}` · top-k: `{metrics.get('top_k')}`\n"
                f"- Retrieval: `{metrics.get('retrieval_mode', 'diversified')}` · "
                f"max `{metrics.get('max_chunks_per_source', 2)}` chunks/URL\n"
                f"- Embedding batches: `{metrics.get('embedding_batches', 0)}` · "
                f"embed time: `{metrics.get('embedding_seconds', 0):.2f}s`\n"
                f"- Chunks indexed: `{chunks}` · duplicates removed: "
                f"`{metrics.get('chunks_deduplicated', 0)}`"
            )
            if metrics.get("rag_error"):
                st.warning(f"RAG fallback used: {metrics['rag_error']}")

        with col_b:
            st.markdown("**Token accounting**")
            st.markdown(
                f"- Raw scraped text: `{metrics.get('raw_scraped_tokens', 0):,}` tokens\n"
                f"- Full-context baseline: `{metrics.get('baseline_context_tokens', 0):,}` tokens\n"
                f"- RAG writer context: `{metrics.get('optimized_context_tokens', 0):,}` tokens\n"
                f"- Saved: `{saved:,}` tokens ({reduction:.1f}%)"
            )

            st.markdown("**Citation / groundedness check**")
            st.markdown(
                f"- References: `{evaluation.get('reference_count', 0)}` · "
                f"verified URLs: `{len(evaluation.get('verified_urls', []))}`\n"
                f"- Citation precision: `{(evaluation.get('citation_precision') or 0) * 100:.0f}%` · "
                f"claim coverage: `{(evaluation.get('claim_coverage') or 0) * 100:.0f}%`\n"
                f"- Retrieved-URL coverage: `{(evaluation.get('retrieved_url_coverage') or 0) * 100:.0f}%` · "
                f"uncited findings: `{evaluation.get('uncited_claim_count', 0)}`\n"
                f"- Check: `{'PASS' if evaluation.get('passed') else 'REVIEW'}`"
            )
            unsupported = evaluation.get("unsupported_urls") or []
            if unsupported:
                st.error("Unsupported reference URLs:\n" + "\n".join(f"- {url}" for url in unsupported))
            for note in evaluation.get("notes") or []:
                st.caption(note)

        st.download_button(
            "⬇  Download metrics (.json)",
            data=json.dumps(
                {"metrics": metrics, "evaluation": evaluation}, indent=2
            ),
            file_name=f"research_metrics_{int(time.time())}.json",
            mime="application/json",
        )


# ── Helper: render a step card ────────────────────────────────────────────────
def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done":    ("✓ DONE",   "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ("results", "running", "done"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else False
st.session_state.setdefault("error", None)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Four specialized AI agents collaborate — searching, scraping, writing,
        and critiquing — to deliver a polished research report on any topic.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        key="topic_input",
        label_visibility="visible",
    )
    run_btn = st.button("⚡  Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Example chips
    st.markdown("""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#605850;letter-spacing:0.1em;">TRY →</span>
    """, unsafe_allow_html=True)
    examples = ["LLM agents 2025", "CRISPR gene editing", "Fusion energy progress"]
    for ex in examples:
        st.markdown(f"""
        <span style="
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:6px;
            padding:0.25rem 0.7rem;
            font-size:0.75rem;
            color:#a09890;
            font-family:'DM Sans',sans-serif;
            cursor:default;
        ">{ex}</span>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)

    r = st.session_state.results
    done = st.session_state.done

    def s(step):
        if not r:
            return "waiting"
        steps = ["search", "reader", "writer", "critic"]
        # figure out which steps are done
        if step in r:
            return "done"
        # which step is running now (first not in r)
        if st.session_state.running:
            for i, k in enumerate(steps):
                if k not in r:
                    return "running" if k == step else "waiting"
        return "waiting"

    step_card("01", "Search Agent",  s("search"), "Gathers recent web information")
    step_card("02", "Reader Agent",  s("reader"), "Scrapes sources & builds ChromaDB index")
    step_card("03", "Writer Chain",  s("writer"), "Retrieves top-k evidence & drafts the report")
    step_card("04", "Critic Chain",  s("critic"), "Reviews, scores & citation-checks the report")


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.error = None
        st.rerun()

if st.session_state.running and not st.session_state.done:
    topic_val = st.session_state.topic_input
    stage_meta = {
        "search": ("🔍", "Search agent"),
        "read": ("📄", "Reader + scraping"),
        "rag": ("🧩", "ChromaDB indexing"),
        "write": ("✍️", "Writer"),
        "critic": ("🧐", "Critic"),
    }

    try:
        with st.status("⚡ Running the 4-stage research pipeline…", expanded=True) as status:
            def on_stage(stage: str, event: str, detail: dict):
                icon, label = stage_meta.get(stage, ("•", stage.title()))
                if event == "running":
                    status.write(f"{icon} **{label}** — running…")
                elif event == "done":
                    if stage == "rag":
                        status.write(
                            f"{icon} **{label}** — {detail.get('chunks', 0)} chunks · "
                            f"top-{detail.get('retrieved', 0)} retrieved via "
                            f"{detail.get('provider', 'n/a')}"
                        )
                    elif stage == "read":
                        status.write(
                            f"{icon} **{label}** — {detail.get('sources', 0)} sources · "
                            f"{detail.get('chars', 0):,} chars"
                        )
                    elif stage == "search":
                        status.write(
                            f"{icon} **{label}** — {detail.get('urls', 0)} candidate URLs"
                        )
                    else:
                        status.write(f"{icon} **{label}** — done")
                elif event == "error":
                    status.write(f"⚠️ **{label}** — {detail.get('error', 'failed')}")

            state = run_research_pipeline(
                topic_val,
                max_sources=4,
                top_k=5,
                verbose=False,
                on_stage=on_stage,
            )
            status.update(label="✓ Research complete", state="complete", expanded=False)

        st.session_state.results = {
            "search": state["search_results"],
            "reader": state["scraped_content"],
            "writer": state["report"],
            "critic": state["feedback"],
            "metrics": state.get("metrics", {}),
            "evaluation": state.get("evaluation", {}),
            "sources": state.get("sources", []),
            "retrieved_chunks": state.get("retrieved_chunks", []),
        }
        st.session_state.error = None
    except Exception as exc:
        st.session_state.results = {}
        st.session_state.error = str(exc)

    st.session_state.running = False
    st.session_state.done = True
    st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if st.session_state.get("error"):
    st.error(f"Pipeline failed: {st.session_state.error}")

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    if r.get("metrics"):
        render_metrics(r["metrics"], r.get("evaluation") or {})

    # Raw outputs in expanders
    if "search" in r:
        with st.expander("🔍 Search Results (raw)", expanded=False):
            st.markdown(f'<div class="result-panel"><div class="result-panel-title">Search Agent Output</div>'
                        f'<div class="result-content">{r["search"]}</div></div>', unsafe_allow_html=True)

    if "reader" in r:
        with st.expander("📄 Scraped Content (raw)", expanded=False):
            st.markdown(f'<div class="result-panel"><div class="result-panel-title">Reader Agent Output</div>'
                        f'<div class="result-content">{r["reader"]}</div></div>', unsafe_allow_html=True)

    if r.get("retrieved_chunks"):
        with st.expander("🔎 Retrieved Evidence (ChromaDB top-k)", expanded=False):
            for index, chunk in enumerate(r["retrieved_chunks"], start=1):
                st.markdown(
                    f"**[{index}] {chunk.get('title') or chunk.get('url') or 'Source'}**  \n"
                    f"`{chunk.get('url') or 'n/a'}` · relevance `{chunk.get('score', 0):.3f}`"
                )
                st.caption(chunk.get("content", "")[:800])

    if r.get("sources"):
        with st.expander("🌐 Sources", expanded=False):
            for source in r["sources"]:
                state_label = "scraped" if source.get("ok") else "failed"
                st.markdown(
                    f"- **{source.get('title') or source.get('url') or 'Untitled'}**  \n"
                    f"  `{source.get('url') or 'n/a'}` · {source.get('char_count', 0):,} chars · {state_label}"
                )

    # Final report
    if "writer" in r:
        st.markdown("""
        <div class="report-panel">
            <div class="panel-label orange">📝 Final Research Report</div>
        """, unsafe_allow_html=True)
        st.markdown(r["writer"])   # render markdown natively
        st.markdown("</div>", unsafe_allow_html=True)

        # Download
        st.download_button(
            label="⬇  Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    # Critic feedback
    if "critic" in r:
        st.markdown("""
        <div class="feedback-panel">
            <div class="panel-label green">🧐 Critic Feedback</div>
        """, unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Powered by LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)