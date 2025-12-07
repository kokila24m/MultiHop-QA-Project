import json
import requests
import streamlit as st

# ======================
# Page config
# ======================

st.set_page_config(
    page_title="NeroNova · Multi-Hop QA with Entailment Verification",
    page_icon="🧠",
    layout="wide",
)

# ======================
# Custom CSS (pale green + aesthetic font)
# ======================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="stApp"] {
        background: linear-gradient(135deg, #e6f7ec 0%, #f2fff8 60%, #ffffff 100%);
        font-family: 'Poppins', sans-serif;
    }

    .neronova-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #184f3d;
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        background: rgba(24, 79, 61, 0.06);
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    }

    .neronova-subtitle {
        font-size: 0.98rem;
        color: #3d6b58;
        margin-top: 0.3rem;
    }

    .neronova-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.7rem;
        border-radius: 999px;
        background: rgba(73, 170, 120, 0.12);
        color: #2a7b50;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 0.3rem;
    }

    .neronova-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.06);
        border: 1px solid rgba(165, 214, 190, 0.7);
        backdrop-filter: blur(10px);
    }

    .neronova-answer {
        font-size: 1.3rem;
        font-weight: 600;
        color: #123629;
    }

    .metric-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        margin-right: 0.3rem;
        margin-bottom: 0.25rem;
        font-size: 0.8rem;
        font-weight: 500;
        background: rgba(24, 79, 61, 0.06);
        color: #184f3d;
    }

    .metric-chip strong {
        margin-right: 0.25rem;
    }

    .metric-chip.good {
        background: rgba(56, 142, 60, 0.14);
        color: #1b5e20;
    }

    .metric-chip.warn {
        background: rgba(255, 193, 7, 0.16);
        color: #795548;
    }

    .retrieval-hop {
        border-radius: 14px;
        padding: 0.65rem 0.8rem;
        background: rgba(230, 247, 236, 0.9);
        border: 1px solid rgba(165, 214, 190, 0.8);
        margin-bottom: 0.4rem;
    }

    .retrieval-hop-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #184f3d;
        margin-bottom: 0.2rem;
    }

    .retrieval-hop-subtitle {
        font-size: 0.82rem;
        color: #366550;
        margin-bottom: 0.2rem;
    }

    .neronova-footer {
        font-size: 0.75rem;
        color: #6b8f7d;
        text-align: right;
        margin-top: 0.7rem;
    }

    /* Hide Streamlit default footer & menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================
# Header
# ======================

col_title, col_badges = st.columns([0.7, 0.3])

with col_title:
    st.markdown(
        """
        <div class="neronova-title">
            NeroNova · Multi-Hop Reasoner
        </div>
        <div class="neronova-subtitle">
            A pale-green, evidence-aware bot that thinks in chains and verifies with logic.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_badges:
    st.markdown(
        """
        <div style="text-align:right;">
            <span class="neronova-badge">🔗 Multi-Hop RAG</span>
            <span class="neronova-badge">✅ Entailment Verified</span>
            <span class="neronova-badge">🧠 LLaMA Powered</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")  # small spacing

# ======================
# Sidebar controls
# ======================

st.sidebar.markdown("### ⚙️ NeroNova Controls")

api_url = st.sidebar.text_input(
    "API Endpoint",
    value="http://localhost:8000/ask",
    help="Your FastAPI /ask endpoint.",
)

top_k = st.sidebar.slider("Top-k documents", min_value=1, max_value=10, value=5, step=1)
chain_length = st.sidebar.slider("Chain length (hops per chain)", min_value=1, max_value=4, value=2, step=1)
num_hops = st.sidebar.slider("Retrieval hops (iterations)", min_value=1, max_value=4, value=2, step=1)

entailment_threshold = st.sidebar.slider(
    "Entailment threshold",
    min_value=0.5,
    max_value=0.95,
    value=0.7,
    step=0.01,
    help="Higher = stricter evidence acceptance.",
)

eval_mode = st.sidebar.checkbox(
    "Eval mode (use gold answers if available)",
    value=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>
    <strong>NeroNova tips</strong><br/>
    • Use detailed, multi-hop style questions.<br/>
    • Experiment with <code>num_hops</code> and <code>chain_length</code>.<br/>
    • Watch how evidence chains and entailment scores change.
    </small>
    """,
    unsafe_allow_html=True,
)

# ======================
# Main input area
# ======================

st.markdown("#### 🧩 Ask NeroNova a multi-hop question")

default_query = (
    "Who is the individual associated with the cryptocurrency industry facing a criminal trial "
    "on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused "
    "by prosecutors of committing fraud for personal gain?"
)

query = st.text_area(
    "Your question",
    value=default_query,
    height=120,
    placeholder="Type a complex, multi-hop question here...",
)

col_btn, col_clear = st.columns([0.2, 0.8])
with col_btn:
    ask_clicked = st.button("✨ Ask NeroNova", type="primary")
with col_clear:
    clear_clicked = st.button("🧹 Clear")

if clear_clicked:
    query = ""
    st.experimental_rerun()

# ======================
# Helper: metrics chips
# ======================

def metric_chip(key: str, value: float) -> str:
    cls = "metric-chip"
    if value >= 0.8:
        cls += " good"
    elif value > 0:
        cls += " warn"
    return f'<span class="{cls}"><strong>{key}</strong>{value:.3f}</span>'


# ======================
# Call backend & render
# ======================

if ask_clicked and query.strip():
    payload = {
        "query": query.strip(),
        "top_k": top_k,
        "chain_length": chain_length,
        "entailment_threshold": entailment_threshold,
        "eval_mode": eval_mode,
        "num_hops": num_hops,
    }

    with st.spinner("NeroNova is traversing documents, building chains, and checking entailment..."):
        try:
            resp = requests.post(api_url, json=payload, timeout=300)
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
            st.stop()

    if resp.status_code != 200:
        st.error(f"API returned status {resp.status_code}: {resp.text}")
        st.stop()

    try:
        data = resp.json()
    except Exception:
        st.error("Could not decode JSON response from API.")
        st.text(resp.text)
        st.stop()

    # ======================
    # Answer card
    # ======================

    st.markdown("### 🧠 NeroNova's Answer")

    with st.container():
        st.markdown('<div class="neronova-card">', unsafe_allow_html=True)

        answer = data.get("predicted_answer", "—")
        gold_answer = data.get("gold_answer", None)
        metrics = data.get("metrics", {})

        st.markdown(
            f"""
            <div class="neronova-answer">
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if gold_answer:
            st.markdown(
                f"""
                <div style="margin-top:0.4rem; font-size:0.9rem; color:#4c7663;">
                    <strong>Gold answer:</strong> <code>{gold_answer}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if metrics:
            chips_html = ""
            for k, v in metrics.items():
                chips_html += metric_chip(k, float(v))
            if chips_html:
                st.markdown(
                    f"""
                    <div style="margin-top:0.6rem;">
                        {chips_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class="neronova-footer">
                NeroNova · multi-hop QA with entailment verification
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ======================
    # Retrieval steps (intermediate thinking)
    # ======================

    retrieval_steps = data.get("retrieval_steps", [])
    if retrieval_steps:
        st.markdown("### 🔍 Retrieval Journey (NeroNova's hops)")

        for step in retrieval_steps:
            hop = step.get("hop", 0)
            queries = step.get("queries", [])
            titles = step.get("retrieved_titles", [])

            with st.expander(f"Hop {hop}: {len(titles)} retrieved docs"):
                st.markdown(
                    f"""
                    <div class="retrieval-hop">
                        <div class="retrieval-hop-title">Hop {hop}</div>
                        <div class="retrieval-hop-subtitle">
                            Queries used at this hop:
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                for q in queries:
                    st.markdown(f"- `{q}`")

                st.markdown(
                    """
                        <div class="retrieval-hop-subtitle" style="margin-top:0.35rem;">
                            Top retrieved titles:
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                if titles:
                    for t in titles:
                        if t:
                            st.markdown(f"- **{t}**")
                else:
                    st.markdown("_No titles available for this hop._")

                st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Evidence chains
    # ======================

    chains = data.get("chains", [])
    if chains:
        st.markdown("### 🧬 Evidence Chains & Entailment")

        for idx, chain in enumerate(chains):
            doc_indices = chain.get("doc_indices", [])
            titles = chain.get("titles", [])
            entail_score = float(chain.get("entailment_score", 0.0))
            supporting = bool(chain.get("supporting", False))
            text = chain.get("concatenated_text", "")

            tag = "✅ Supporting chain" if supporting else "⚪ Candidate chain"
            color = "#1b5e20" if supporting else "#555555"
            score_str = f"{entail_score:.3f}"

            header_html = f"""
            <span style="color:{color}; font-weight:600;">{tag}</span>
            <span style="font-size:0.8rem; color:#607d8b; margin-left:0.5rem;">
                entailment = {score_str}
            </span>
            """

            with st.expander(f"Chain {idx+1}: {header_html}", expanded=supporting):
                st.markdown(
                    f"""
                    <div style="font-size:0.88rem; color:#375a4a; margin-bottom:0.35rem;">
                        <strong>Documents:</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if titles:
                    for d_i, t in zip(doc_indices, titles):
                        st.markdown(f"- **[{d_i}]** {t}")
                else:
                    st.markdown("_No titles available for this chain._")

                st.markdown(
                    """
                    <div style="font-size:0.88rem; color:#375a4a; margin-top:0.5rem; margin-bottom:0.2rem;">
                        <strong>Concatenated evidence text:</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show long text in a text area for easy scroll
                st.text_area(
                    label="",
                    value=text,
                    height=180,
                    key=f"chain_text_{idx}",
                )

else:
    st.markdown(
        """
        <div style="margin-top:1.4rem; font-size:0.9rem; color:#567765;">
            👋 Type a question above and click <strong>“Ask NeroNova”</strong> to see multi-hop retrieval,
            evidence chains, and entailment-based verification in action.
        </div>
        """,
        unsafe_allow_html=True,
    )
