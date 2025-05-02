# app.py
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ─── 1) Load vectorizer & model ───────────────────────────────────────────────
BASE = Path(__file__).parent
with open(BASE/"models"/"tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open(BASE/"models"/"logreg.pkl", "rb") as f:
    model = pickle.load(f)

# ─── 2) Load & split the raw SMS data (for metrics) ─────────────────────────────
data_file = BASE.parent/"data"/"raw"/"sms.tsv"
df = pd.read_csv(
    data_file,
    sep="\t",
    header=None,
    names=["label","message"],
    encoding="utf-8"
)
df = df.drop_duplicates().reset_index(drop=True)
df["label_num"] = df["label"].map({"ham":0, "spam":1})
X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
X_test_tfidf = vectorizer.transform(X_test)
y_pred       = model.predict(X_test_tfidf)
y_prob       = model.predict_proba(X_test_tfidf)[:,1]

# compute metrics once
acc      = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc  = auc(fpr, tpr)

# ─── 3) Streamlit page config & title ───────────────────────────────────────────
st.set_page_config(page_title="SMS Spam Detector", layout="wide")
st.title("📱 SMS Spam Detector")
st.markdown(
    "Welcome! Type or paste an SMS message below and click **Classify**  \n"
    "to see if it’s **Ham** (legitimate) or **Spam**."
)

# ─── 4) Sidebar: show model performance ─────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.markdown(f"**Accuracy:** {acc:.2%}")

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.matshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center")
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["Ham","Spam"]); ax_cm.set_yticklabels(["Ham","Spam"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC Curve
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0,1],[0,1], "k--", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# ─── 5) Main layout: two columns ────────────────────────────────────────────────
col1, col2 = st.columns([2,1])

# —— Single‐message classification
with col1:
    user_text = st.text_area("✏️ Enter your SMS here", height=150)
    if st.button("🔍 Classify"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            X_new  = vectorizer.transform([user_text])
            pred   = model.predict(X_new)[0]
            prob   = model.predict_proba(X_new)[0, pred]
            label  = "✅ Ham" if pred == 0 else "🚫 Spam"
            st.markdown(f"### {label}")
            st.write(f"Prediction confidence: **{prob:.1%}**")

# —— Batch upload for many messages
with col2:
    st.subheader("📂 Bulk classify")
    st.markdown("Upload a **.csv** or **.tsv** file with a `message` column.")
    uploaded = st.file_uploader("", type=["csv","tsv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded, sep=None, engine="python")
        if "message" not in df_batch.columns:
            st.error("Your file must contain a `message` column.")
        else:
            Xb    = vectorizer.transform(df_batch["message"])
            preds = model.predict(Xb)
            df_batch["prediction"] = np.where(preds==0, "Ham", "Spam")
            st.dataframe(df_batch)

