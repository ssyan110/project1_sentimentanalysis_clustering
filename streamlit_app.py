# streamlit_app.py
# ──────────────────────────────────────────────────────────────────────
# Dash-style Streamlit app for ITviec review exploration
# (expects every artefact under  outputs/  as written in §5 & §6)
# ──────────────────────────────────────────────────────────────────────
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA

sns.set(style="whitegrid")
OUT = "outputs"                     # ← single place to change if needed

# ╭──────────────────────  DATA LOAD (cached)  ─────────────────────────╮
@st.cache_data
def load_data():
    reviews_df = pd.read_csv(f"{OUT}/clean_reviews.csv")
    company_df = pd.read_csv(f"{OUT}/company_clusters_with_topics.csv")
    model_df   = pd.read_csv(f"{OUT}/model_results.csv")

    def _try_json(fn):
        try:
            with open(f"{OUT}/{fn}", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    cluster_terms  = _try_json("cluster_terms.json")
    lda_topics     = _try_json("lda_topics.json")
    cluster_labels = _try_json("cluster_labels.json")   # NEW – pretty names
    return reviews_df, company_df, model_df, cluster_terms, lda_topics, cluster_labels

reviews_df, company_df, model_df, cluster_terms, lda_topics, cluster_labels = load_data()

# ╭───────────────────  SIDEBAR NAVIGATION  ────────────────────────────╮
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    [
        "Data Overview",
        "Sentiment Analysis",
        "Clustering",
        "Topic Modeling",
        "Model Results",
        "Company Insight",
    ],
)

# ╭───────────────────────  PAGE 1 – OVERVIEW  ─────────────────────────╮
if page == "Data Overview":
    st.title("ITviec Reviews – Data Overview")
    st.markdown(f"**Total reviews:** {len(reviews_df):,}")
    st.markdown(f"**Unique companies:** {company_df['id'].nunique():,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Star Rating Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Rating", data=reviews_df, palette="viridis", ax=ax)
        ax.set_xlabel("Rating (1–5)")
        st.pyplot(fig)

    with col2:
        st.subheader("Recommend? (Yes / No)")
        fig, ax = plt.subplots()
        order = reviews_df["Recommend?"].value_counts().index
        sns.countplot(x="Recommend?", data=reviews_df, order=order, palette="Set2", ax=ax)
        st.pyplot(fig)

    st.subheader("Review Length (tokens)")
    fig, ax = plt.subplots()
    reviews_df["clean_review"].str.split().str.len().hist(bins=30, ax=ax)
    ax.set_xlabel("Tokens"); ax.set_ylabel("Count")
    st.pyplot(fig)

# ╭──────────────────────  PAGE 2 – SENTIMENT  ─────────────────────────╮
elif page == "Sentiment Analysis":
    st.title("Lexicon-Based Sentiment Analysis")
    counts = (
        reviews_df["sentiment"]
        .value_counts().reindex(["positive", "neutral", "negative"])
        .fillna(0).astype(int)
    )
    st.bar_chart(counts)
    st.write("**Proportions**")
    st.write((counts / counts.sum()).rename("proportion").to_frame())

    st.subheader("Random Examples")
    rng = np.random.default_rng(42)
    for sent in ["positive", "neutral", "negative"]:
        st.markdown(f"**{sent.title()}**")
        sample = reviews_df.loc[reviews_df["sentiment"] == sent, "review"]
        for txt in rng.choice(sample.values, size=min(3, len(sample)), replace=False):
            st.write(f"> {txt}")
        st.write("---")

# ╭──────────────────────  PAGE 3 – CLUSTERING  ────────────────────────╮
elif page == "Clustering":
    st.title("Company Clustering")

    algo_cols = sorted(
        [c for c in company_df.columns if c.startswith("cluster_") or c.endswith("_kmeans")]
    )
    if not algo_cols:
        st.warning("No clustering columns found in company_clusters_with_topics.csv")
        st.stop()

    default_algo = "cluster_kmeans" if "cluster_kmeans" in algo_cols else algo_cols[0]
    algo = st.selectbox("Choose clustering algorithm", algo_cols,
                        index=algo_cols.index(default_algo))

    # friendly display series
    disp_series = company_df[algo]
    if algo == "lda_kmeans" and cluster_labels:
        disp_series = disp_series.map(lambda x: cluster_labels.get(str(int(x)), f"Cluster {x}"))

    st.subheader("Cluster Sizes")
    st.bar_chart(disp_series.value_counts().sort_index())

    # ─── PCA scatter of LDA topic space ───────────────────────────────
    st.subheader("PCA of LDA topic space")
    pca_png = f"{OUT}/lda_topics_pca.png"
    if os.path.exists(pca_png):
        st.image(pca_png, caption="PCA of LDA topics (pre-computed)")
    elif {"ldaPC1", "ldaPC2"}.issubset(company_df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        hue_col = "lda_kmeans" if "lda_kmeans" in company_df.columns else algo
        sns.scatterplot(data=company_df, x="ldaPC1", y="ldaPC2",
                        hue=hue_col, palette="Set2", s=70, alpha=0.85, ax=ax)
        ax.set_title("PCA of LDA topics")
        st.pyplot(fig)
    else:
        st.info("No pre-computed PCA available.")

    # ─── Signature terms ──────────────────────────────────────────────
    st.subheader("Signature Terms (lda_kmeans)")
    term_block = cluster_terms.get("lda_kmeans", {})
    if isinstance(term_block, dict) and term_block:
        for cid, words in term_block.items():
            label = cluster_labels.get(cid, f"Cluster {cid}")
            st.markdown(f"**{label}:** {', '.join(words)}")
    else:
        st.info("No signature-term dictionary available.")

# ╭──────────────────────  PAGE 4 – TOPIC MODEL  ───────────────────────╮
elif page == "Topic Modeling":
    st.title("LDA Topic Modeling")
    if not lda_topics:
        st.warning("lda_topics.json not found."); st.stop()

    st.subheader("Top Words per Topic")
    for topic, words in lda_topics.items():
        st.markdown(f"**{topic}:** {', '.join(words)}")

    st.subheader("Company Counts per Topic")
    st.bar_chart(company_df["lda_topic"].value_counts().sort_index())

# ╭──────────────────────  PAGE 5 – MODEL RESULTS  ─────────────────────╮
elif page == "Model Results":
    st.title("Sentiment-Model Leaderboard")
    st.dataframe(model_df.style.format({"accuracy": "{:.3f}", "f1": "{:.3f}"}))

# ╭──────────────────────  PAGE 6 – COMPANY INSIGHT  ───────────────────╮
elif page == "Company Insight":
    st.title("🔍 Company-Level Insight")

    ids = company_df["id"].sort_values().unique()
    comp_id = st.number_input("Enter company ID",
                              min_value=int(ids.min()),
                              max_value=int(ids.max()),
                              value=int(ids[0]), step=1)

    if comp_id not in ids:
        st.error("Company ID not found."); st.stop()

    meta      = company_df.loc[company_df["id"] == comp_id].iloc[0]
    comp_revs = reviews_df[reviews_df["id"] == comp_id]

    # friendly cluster name
    cl_id   = str(int(meta.get("lda_kmeans", -1)))
    cl_name = cluster_labels.get(cl_id, f"Cluster {cl_id}")

    st.header(f"{meta.CompanyName} (ID {comp_id})")
    st.markdown(f"- Reviews analysed: **{len(comp_revs)}**")
    st.markdown(f"- LDA topic: **{int(meta.lda_topic)}**")
    if "lda_kmeans" in meta.index:
        st.markdown(f"- Topic-KMeans cluster: **{cl_name}**")

    st.subheader("Sentiment Distribution")
    sent_cnt = (
        comp_revs["sentiment"]
        .value_counts().reindex(["positive", "neutral", "negative"])
        .fillna(0).astype(int)
    )
    st.bar_chart(sent_cnt)

    st.subheader("Most Frequent Words")
    top_tokens = (
        pd.Series(" ".join(comp_revs["clean_review"]).split())
        .value_counts().head(15)
        .rename_axis("word").reset_index(name="freq")
    )
    st.table(top_tokens)

    st.subheader("Signature Terms")
    if cl_id in cluster_terms.get("lda_kmeans", {}):
        st.markdown(f"**{cl_name}:** {', '.join(cluster_terms['lda_kmeans'][cl_id])}")
    else:
        tkey = f"topic_{int(meta.lda_topic)}"
        if tkey in lda_topics:
            st.markdown(f"**{tkey}:** {', '.join(lda_topics[tkey])}")
        else:
            st.info("No signature terms available.")
