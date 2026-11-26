# from pathlib import Path
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st

# BASEDIR = Path(__file__).resolve().parent
# DATADIR = BASEDIR / "Ag_news_Dataset"

st.set_page_config(page_title="AG News Encoder", layout="wide")

LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


@st.cache_resource
def load_encoder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

"""
@st.cache_data
def load_data(filename: str, limit: int) -> pl.DataFrame:
    file_path = DATADIR / filename
    return (
        pl.scan_csv(file_path)
        .limit(limit)
        .with_columns(
            (pl.col("Class Index") - 1).alias("label"),
            (
                pl.col("Title").fill_null("")
                + " - "
                + pl.col("Description").fill_null("")
            ).alias("text"),
        )
        .with_columns(pl.col("label").replace_strict(LABELS).alias("category"))
        .collect()
    )
"""

@st.cache_data
def load_data(split: str, limit: int) -> pl.DataFrame:
    # split should be "train" or "test"
    hf_path = f"hf://datasets/fancyzhx/ag_news/data/{split}-00000-of-00001.parquet"
    return (
        pl.scan_parquet(hf_path)
        .limit(limit)
        .with_columns(
            pl.col("label").replace_strict(LABELS).alias("category")
        )
        .collect()
    )

@st.cache_data(ttl=3600)
def generate_embeddings(texts: list[str]):
    encoder = load_encoder()
    return encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

@st.cache_resource
def train_classifier(X_train, y_train) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf



st.title("AG News Article Categorizer")

with st.sidebar:
    st.header("Settings")
    filename = st.text_input("Dataset Filename", value="train.csv")
    sample_limit = st.slider("Number of Articles", min_value=500, max_value=5000, value=1500, step=500)
    test_size = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

try:
    df = load_data(filename, sample_limit)
except Exception as e:
    st.error(f"Coudl not load dataset from '{DATADIR / filename}' : {e}")
    st.stop()

with st.spinner("Encoding articles with BAAI/bge-small-en-v1.5..."):
    embeddings = generate_embeddings(df["text"].to_list())
    labels = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=test_size, random_state=101, stratify=labels,
)

clf = train_classifier(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns ([1, 2])
with col1:
    st.metric("Test Accuracy", f"{acc:.2%}")
    st.write("**Class Breakdown:**")
    st.dataframe(
        df.group_by("category").len().sort("len", descending=True),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.subheader("Sample Data Preview")
    st.dataframe(df.select(["category", "Title", "Description"]).head(5), use_container_width=True)

st.divider()
st.subheader("Categorize a New Article")
user_title = st.text_input("Headline", placeholder="Apple unveils new M-series chip for MacBook Pro.")
user_desc = st.text_area("Description", placeholder="The tech giant announced faster neural engine performance.")

if st.button("Predict Category", type="primary"):
    if user_title.strip() or user_desc.strip():
        combined_text = f"{user_title.strip()} - {user_desc.strip()}"
        encoder = load_encoder()
        vec = encoder.encode([combined_text], normalize_embeddings=True)

        pred_idx = int(clf.predict(vec)[0])
        probs = clf.predict_proba(vec)[0]

        st.success(f"**Predicted Category:** {LABELS[pred_idx]}")

        prob_df = pl.DataFrame({
            "Category": list(LABELS.values()),
            "Confidence": [float(p) for p in probs]
        }).sort("Confidence", descending=True)

        st.bar_chart(prob_df, x="Category", y="Confidnece")
    else:
        st.warning("Please enter a headline or description of some sort first")