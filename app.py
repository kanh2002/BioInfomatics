import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from xgboost import XGBClassifier

st.set_page_config(page_title="DNA Model Studio", page_icon="🧬", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
BENCHMARK_PATH = MODEL_DIR / "benchmark_results.csv"
DNA_TO_INDEX = {"A": 1, "T": 2, "C": 3, "G": 4}


class DeepTextCNN(nn.Module):
    """
    Compatible with the improved checkpoint:
    embedding: [5, 64]
    convs: 4 kernels = (3, 5, 7, 11)
    num_filters: 128
    classifier: Sequential Linear-ReLU-Dropout-Linear
    """
    def __init__(
        self,
        vocab_size=5,
        embed_dim=64,
        num_filters=128,
        kernel_sizes=(3, 5, 7, 11),
        dropout=0.40,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.bn = nn.BatchNorm1d(num_filters * len(kernel_sizes))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        feature_maps = []
        for conv in self.convs:
            activated = torch.relu(conv(x))
            max_pool = torch.max(activated, dim=2).values
            avg_pool = torch.mean(activated, dim=2)
            pooled = 0.5 * max_pool + 0.5 * avg_pool
            feature_maps.append(pooled)

        features = torch.cat(feature_maps, dim=1)
        features = self.bn(features)
        features = self.dropout(features)
        return self.classifier(features)


class DeepTextCNNLegacy(nn.Module):
    """
    Compatible with the old checkpoint:
    embedding: [5, 32]
    convs: 3 kernels = (3, 5, 7)
    num_filters: 96
    classifier: Linear
    """
    def __init__(self, vocab_size=5, embed_dim=32, num_filters=96, kernel_sizes=(3, 5, 7), dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        feature_maps = []
        for conv in self.convs:
            activated = torch.relu(conv(x))
            pooled = torch.max(activated, dim=2).values
            feature_maps.append(pooled)
        features = torch.cat(feature_maps, dim=1)
        features = self.dropout(features)
        return self.classifier(features)

def display_model_name(name):
    name = str(name)

    if "DeepTextCNN" in name:
        return "DeepTextCNN"
    if "CNNBiLSTM" in name:
        return "CNNBiLSTM"


    return name
def build_deep_textcnn_from_checkpoint(checkpoint):
    """
    Auto-detect whether checkpoint is old or improved, then build the correct architecture.
    """
    state = checkpoint["state_dict"]

    embed_dim = state["embedding.weight"].shape[1]
    num_filters = state["convs.0.weight"].shape[0]
    num_convs = len([k for k in state.keys() if k.startswith("convs.") and k.endswith(".weight")])

    has_bn = any(k.startswith("bn.") for k in state.keys())
    has_sequential_classifier = any(k.startswith("classifier.0.") for k in state.keys())

    if embed_dim == 64 and num_filters == 128 and num_convs == 4 and has_bn and has_sequential_classifier:
        return DeepTextCNN()

    if embed_dim == 32 and num_filters == 96 and num_convs == 3 and not has_bn:
        return DeepTextCNNLegacy()

    raise RuntimeError(
        "Unknown DeepTextCNN checkpoint architecture. "
        f"Detected embed_dim={embed_dim}, num_filters={num_filters}, "
        f"num_convs={num_convs}, has_bn={has_bn}, "
        f"has_sequential_classifier={has_sequential_classifier}."
    )


def clean_sequence(seq):
    return "".join(ch for ch in seq.upper() if ch in "ATCG")


def encode_sequence(seq, max_len):
    arr = np.zeros((1, max_len), dtype=np.int64)
    seq_ids = [DNA_TO_INDEX.get(ch, 0) for ch in seq[:max_len]]
    arr[0, : len(seq_ids)] = seq_ids
    return torch.tensor(arr, dtype=torch.long)


@st.cache_resource(show_spinner="Đang tải các mô hình (Loading models)...")
def load_assets(vectorizer_mtime):
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    required = [vectorizer_path]
    missing_files = [str(path) for path in required if not path.exists()]
    if missing_files:
        missing_list = "\n".join(missing_files)
        raise FileNotFoundError(
            "Missing core model artifacts. Run: python DNA_Cls.py\nMissing files:\n"
            f"{missing_list}"
        )

    vectorizer = joblib.load(vectorizer_path)
    # Lấy 1 sequence mẫu để kiểm tra lỗi feature shape mismatch
    x_probe = vectorizer.transform(["ATCGATCGATCG"])

    models = {}
    skipped_models = []

    def maybe_add_model(model_name, model_obj):
        try:
            # Kiểm tra xem có dự đoán được không để bẫy lỗi ValueError mismatch features
            if hasattr(model_obj, "predict_proba"):
                _ = model_obj.predict_proba(x_probe)
            else:
                _ = model_obj.predict(x_probe)
            models[model_name] = model_obj
        except Exception as exc:
            skipped_models.append(f"{model_name}: {exc}")

    nusvc_candidates = [
        ("NuSVC", MODEL_DIR / "NuSVC_RBF.joblib"),
        ("NuSVC", MODEL_DIR / "nusvc_diabetes_model.joblib"),
    ]
    for name, path in nusvc_candidates:
        if path.exists() and name not in models:
            maybe_add_model(name, joblib.load(path))

    xgb_joblib = MODEL_DIR / "XGBoost.joblib"
    xgb_json = MODEL_DIR / "xgb_diabetes_model.json"
    if xgb_joblib.exists():
        maybe_add_model("XGBoost", joblib.load(xgb_joblib))
    elif xgb_json.exists():
        xgb = XGBClassifier()
        xgb.load_model(str(xgb_json))
        maybe_add_model("XGBoost", xgb)

    optional_models = {
        "RandomForest": MODEL_DIR / "rf_diabetes_model.joblib",
        "MLP": MODEL_DIR / "mlp_diabetes_model.joblib",
        "ExtraTrees": MODEL_DIR / "ExtraTrees.joblib",
        "LinearSVC_Calibrated": MODEL_DIR / "LinearSVC_Calibrated.joblib",
        "MLP_SVD": MODEL_DIR / "MLP_SVD.joblib",
    }
    for name, file_path in optional_models.items():
        if file_path.exists():
            maybe_add_model(name, joblib.load(file_path))

    deep_path = MODEL_DIR / "deep_textcnn_improved.pt"
    if not deep_path.exists():
        deep_path = MODEL_DIR / "deep_textcnn_model.pt"
    if deep_path.exists():
        deep_checkpoint = torch.load(deep_path, map_location="cpu")
        deep_model = build_deep_textcnn_from_checkpoint(deep_checkpoint)
        deep_model.load_state_dict(deep_checkpoint["state_dict"])
        deep_model.eval()
        model_name="DeepTextCNN_Improved"
        models[model_name] = {
            "type": "deep_textcnn_improved",
            "model": deep_model,
            "max_len": int(deep_checkpoint["max_len"]),
        }

    if not models:
        raise FileNotFoundError(
            "No compatible models found for the current tfidf_vectorizer.joblib. "
            "Please run: python DNA_Cls.py"
        )

    return vectorizer, models, skipped_models


@st.cache_data
def load_benchmark_table():
    if not BENCHMARK_PATH.exists():
        return None
    table = pd.read_csv(BENCHMARK_PATH)
    return table.sort_values(by=["roc_auc", "f1"], ascending=False)


def infer_all_models(vectorizer, models, dna_seq):
    x_new = vectorizer.transform([dna_seq])
    results = []

    for name, model in models.items():

        # Deep learning models được lưu dạng dict
        if isinstance(model, dict):
            model_type = model.get("type", "")

            if model_type in ["deep_textcnn", "deep_textcnn_improved", "cnn_bilstm"]:
                encoded = encode_sequence(dna_seq, model["max_len"])

                with torch.no_grad():
                    logits = model["model"](encoded)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                print(f"Skip unknown deep model type: {name} - {model_type}")
                continue

        # Classical ML models
        else:
            probs = model.predict_proba(x_new)[0]

        diabetic_prob = float(probs[1])
        pred_label = "Diabetic" if int(np.argmax(probs)) == 1 else "Non-diabetic"
        confidence = float(np.max(probs)) * 100.0

        results.append(
            {
                "Model": name,
                "Prediction": pred_label,
                "Diabetic_Prob": diabetic_prob,
                "Confidence": confidence,
            }
        )

    return pd.DataFrame(results).sort_values(by="Diabetic_Prob", ascending=False)

def render_score_card(model_name, row):
    risk = row["Diabetic_Prob"]
    risk_pct = risk * 100
    st.markdown(f"**{model_name}**")
    
    if risk >= 0.5:
        st.error(f"⚠️ {row['Prediction']} ({risk_pct:.1f}%)")
    else:
        st.success(f"✅ {row['Prediction']} ({risk_pct:.1f}%)")
    
    st.progress(float(risk))
    st.caption(f"Confidence: {row['Confidence']:.2f}%")


def risk_label(prob, threshold):
    if prob >= threshold + 0.15:
        return "High"
    if prob >= threshold:
        return "Medium"
    return "Low"


def risk_row_style(row):
    level = row.get("Risk_Level", "Low")
    if level == "High":
        color = "background-color: rgba(255, 0, 0, 0.15); color: #ff4b4b;"
    elif level == "Medium":
        color = "background-color: rgba(255, 165, 0, 0.15); color: #ffa500;"
    else:
        color = "background-color: rgba(0, 128, 0, 0.15); color: #00ea00;"
    return [color] * len(row)


def compute_final_decision(result_df, threshold=0.5):
    probs = result_df["Diabetic_Prob"].astype(float).values
    confidences = result_df["Confidence"].astype(float).values / 100.0

    weights = np.clip(confidences, 0.50, 1.00)
    weights = weights / weights.sum()

    final_prob = float(np.sum(probs * weights))
    mean_prob = float(np.mean(probs))
    std_prob = float(np.std(probs))

    vote_positive = int((probs >= threshold).sum())
    vote_negative = int((probs < threshold).sum())
    n_models = len(result_df)
    final_pred = "Diabetic" if final_prob >= threshold else "Non-diabetic"

    consensus_ratio = max(vote_positive, vote_negative) / max(n_models, 1)

    if consensus_ratio >= 0.80 and std_prob <= 0.15:
        reliability = "High"
        reliability_msg = "Các mô hình khá đồng thuận và độ dao động xác suất thấp."
    elif consensus_ratio >= 0.60 and std_prob <= 0.25:
        reliability = "Medium"
        reliability_msg = "Các mô hình tương đối đồng thuận, nhưng vẫn có dao động nhất định."
    else:
        reliability = "Low"
        reliability_msg = "Các mô hình chưa đồng thuận rõ ràng; nên kiểm tra thêm hoặc dùng dữ liệu bổ sung."

    return {
        "final_prob": final_prob,
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "final_pred": final_pred,
        "vote_positive": vote_positive,
        "vote_negative": vote_negative,
        "n_models": n_models,
        "consensus_ratio": consensus_ratio,
        "reliability": reliability,
        "reliability_msg": reliability_msg,
    }


def render_final_decision_panel(decision, threshold):
    st.subheader("🧠 Final Ensemble Decision")

    prob = decision["final_prob"]
    pred = decision["final_pred"]

    if pred == "Diabetic":
        st.error(f"### Final Prediction: {pred} — Risk {prob*100:.2f}%")
    else:
        st.success(f"### Final Prediction: {pred} — Risk {prob*100:.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weighted Risk", f"{prob*100:.2f}%")
    c2.metric("Mean Risk", f"{decision['mean_prob']*100:.2f}%")
    c3.metric("Risk Std", f"{decision['std_prob']*100:.2f}%")
    c4.metric("Votes", f"{decision['vote_positive']}/{decision['n_models']}", f"> threshold {threshold:.2f}", delta_color="inverse")

    if decision["reliability"] == "High":
        st.success(f"Reliability: **{decision['reliability']}** — {decision['reliability_msg']}")
    elif decision["reliability"] == "Medium":
        st.warning(f"Reliability: **{decision['reliability']}** — {decision['reliability_msg']}")
    else:
        st.error(f"Reliability: **{decision['reliability']}** — {decision['reliability_msg']}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Final Diabetes Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if pred == "Diabetic" else "green"},
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(0, 180, 0, 0.18)"},
                {"range": [threshold * 100, min(100, (threshold + 0.15) * 100)], "color": "rgba(255, 165, 0, 0.20)"},
                {"range": [min(100, (threshold + 0.15) * 100), 100], "color": "rgba(255, 0, 0, 0.20)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def get_feature_names(vectorizer):
    try:
        return np.array(vectorizer.get_feature_names_out())
    except Exception:
        return np.array(vectorizer.get_feature_names())


def explain_tfidf_kmers(vectorizer, model, dna_seq, top_n=15):
    x = vectorizer.transform([dna_seq])
    feature_names = get_feature_names(vectorizer)

    x_arr = x.toarray()[0]
    nonzero = np.where(x_arr > 0)[0]
    if len(nonzero) == 0:
        return pd.DataFrame(columns=["kmer", "tfidf", "model_weight", "local_score"])

    model_weight = np.ones_like(x_arr)

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        if len(imp) == len(model_weight):
            model_weight = imp
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        if len(coef) == len(model_weight):
            model_weight = coef
    else:
        return pd.DataFrame(columns=["kmer", "tfidf", "model_weight", "local_score"])

    local_score = x_arr * model_weight
    idx = nonzero[np.argsort(np.abs(local_score[nonzero]))[::-1][:top_n]]

    out = pd.DataFrame({
        "kmer": feature_names[idx],
        "tfidf": x_arr[idx],
        "model_weight": model_weight[idx],
        "local_score": local_score[idx],
    })

    return out.sort_values("local_score", key=lambda s: np.abs(s), ascending=False)


def render_kmer_explanation(vectorizer, models, dna_seq, top_n=15):
    st.subheader("🧬 Explainability: Top k-mer / motif đóng góp")

    explainable = {}
    for name, model in models.items():
        if isinstance(model, dict):
            continue
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            explainable[name] = model

    if not explainable:
        st.info("Không tìm thấy model TF-IDF có feature_importances_ hoặc coef_ để giải thích k-mer.")
        return

    selected_model = st.selectbox("Chọn model để giải thích k-mer", list(explainable.keys()), key="kmer_explain_model")

    exp_df = explain_tfidf_kmers(vectorizer, explainable[selected_model], dna_seq, top_n=top_n)

    if exp_df.empty:
        st.info("Model này không hỗ trợ giải thích trực tiếp theo k-mer hoặc sequence không có k-mer khớp.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(
            exp_df.assign(
                tfidf=exp_df["tfidf"].round(5),
                model_weight=exp_df["model_weight"].round(5),
                local_score=exp_df["local_score"].round(5),
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        fig = px.bar(
            exp_df.sort_values("local_score"),
            x="local_score",
            y="kmer",
            orientation="h",
            title=f"Top local k-mer contributions — {selected_model}",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("local_score = TF-IDF(k-mer) × model_weight.")


def compute_deep_saliency(deep_model, dna_seq, max_len):
    model = deep_model
    model.eval()

    encoded = encode_sequence(dna_seq, max_len)

    acts = {}

    def forward_hook(module, inp, out):
        acts["emb"] = out
        if out.requires_grad:
            out.retain_grad()

    handle = model.embedding.register_forward_hook(forward_hook)

    try:
        with torch.enable_grad():
            logits = model(encoded)
            prob = torch.softmax(logits, dim=1)[0, 1]

            model.zero_grad(set_to_none=True)
            prob.backward()

            emb = acts["emb"]
            grad = emb.grad

            if grad is None:
                return np.zeros(min(len(dna_seq), max_len))

            sal = grad.abs().sum(dim=2).detach().cpu().numpy()[0]

    finally:
        handle.remove()

    seq_len = min(len(dna_seq), max_len)
    sal = sal[:seq_len]

    if sal.max() > 0:
        sal = sal / sal.max()

    return sal
def find_important_regions(saliency, window=20, top_n=5):
    if len(saliency) == 0:
        return []

    scores = []
    for start in range(0, max(1, len(saliency) - window + 1)):
        end = min(len(saliency), start + window)
        scores.append((start, end, float(np.mean(saliency[start:end]))))

    scores = sorted(scores, key=lambda x: x[2], reverse=True)

    selected = []
    used = np.zeros(len(saliency), dtype=bool)

    for start, end, score in scores:
        overlap = used[start:end].mean() if end > start else 1
        if overlap < 0.30:
            selected.append((start, end, score))
            used[start:end] = True
        if len(selected) >= top_n:
            break

    return selected


def render_deep_saliency(models, dna_seq):
    st.subheader("🔥 DeepTextCNN Saliency Map")

    deep_key = None
    for key, value in models.items():
        if isinstance(value, dict) and "DeepTextCNN" in key:
            deep_key = key
            break

    if deep_key is None:
        st.info("Không tìm thấy DeepTextCNN để tạo saliency map.")
        return

    deep_pack = models[deep_key]


    model = deep_pack["model"]
    max_len = deep_pack["max_len"]

    if len(dna_seq) < 5:
        st.warning("Sequence quá ngắn để vẽ saliency.")
        return

    with st.spinner("Đang tính saliency map..."):
        sal = compute_deep_saliency(model, dna_seq, max_len)

    regions = find_important_regions(sal, window=min(30, max(5, len(sal)//10)), top_n=5)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(sal))), y=sal, mode="lines", name="Saliency"))

    for start, end, score in regions:
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.18, line_width=0)

    fig.update_layout(
        title="Gradient saliency over DNA positions",
        xaxis_title="DNA position",
        yaxis_title="Normalized saliency",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    region_df = pd.DataFrame([
        {
            "region": f"{start}-{end}",
            "mean_saliency": round(score, 4),
            "sequence_fragment": dna_seq[start:end],
        }
        for start, end, score in regions
    ])

    st.markdown("**Vùng DNA quan trọng nhất theo DeepTextCNN:**")
    st.dataframe(region_df, use_container_width=True, hide_index=True)


def render_reliability_analysis(result_df, decision):
    st.subheader("🛡️ Reliability / Robustness Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model disagreement", f"{decision['std_prob']*100:.2f}%")
    c2.metric("Consensus ratio", f"{decision['consensus_ratio']*100:.1f}%")
    c3.metric("Reliability", decision["reliability"])

    fig = px.box(result_df, y="Diabetic_Prob", points="all", title="Distribution of model risk probabilities")
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if decision["reliability"] == "Low":
        st.error("Cảnh báo: Các mô hình bất đồng đáng kể. Không nên diễn giải prediction này như một kết luận chắc chắn.")
    elif decision["reliability"] == "Medium":
        st.warning("Prediction có độ tin cậy trung bình. Nên xem thêm explainability và kiểm tra sequence đầu vào.")
    else:
        st.success("Prediction ổn định giữa các mô hình.")


st.title("Phân tích Sinh tin: DNA Model Studio 🧬")
st.markdown("Hệ thống đánh giá đa mô hình Học máy & Học sâu trên chuỗi DNA liên quan bệnh đái tháo đường.", unsafe_allow_html=True)
st.divider()

if not (MODEL_DIR / "tfidf_vectorizer.joblib").exists():
    st.error("Chưa có mô hình nào được huấn luyện. Vui lòng chạy `python DNA_Cls.py` trước!")
    st.stop()

try:
    vectorizer_mtime = os.path.getmtime(MODEL_DIR / "tfidf_vectorizer.joblib")
    vectorizer, models, skipped_models = load_assets(vectorizer_mtime)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Chạy huấn luyện lần đầu: `python DNA_Cls.py`")
    st.stop()

with st.sidebar:
    st.header("Control Panel")
    threshold = st.slider("Risk threshold", 0.2, 0.8, 0.5, 0.05)
    sort_mode = st.selectbox(
        "Sort results by",
        ["Diabetic probability", "Confidence", "Model name"],
        index=0,
    )
    preset = st.selectbox(
        "Quick sample",
        ["None", "Short sample", "Medium sample", "Long sample"],
        index=0,
    )


tab_predict, tab_explain, tab_leaderboard, tab_guidance = st.tabs(["🧬 Predict (Dự đoán)", "🔍 Explainability", "🏆 Leaderboard (Bảng xếp hạng)", "📚 Guidance (Hướng dẫn)"])


if "last_result_df" not in st.session_state:
    st.session_state["last_result_df"] = None
if "last_cleaned_seq" not in st.session_state:
    st.session_state["last_cleaned_seq"] = None
if "last_decision" not in st.session_state:
    st.session_state["last_decision"] = None

with tab_predict:
    samples = {
        "None": "ATCGATCGATCGATCGATCG",
        "Short sample": "ATCGATCGGCTAATCGATCGATCG",
        "Medium sample": "ATCG" * 50,
        "Long sample": "ATCGGCTA" * 140,
    }
    default_seq = samples.get(preset, samples["None"])
    dna_input = st.text_area("Nhập chuỗi DNA để phân tích (A/T/C/G)", value=default_seq, height=120)

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("Phân tích Ngân hàng Gen 🚀", use_container_width=True, type="primary")
    with col_info:
        st.caption("DNA sequences are cleaned automatically. Only [A, T, C, G] characters are processed.")

    if run_btn:
        cleaned = clean_sequence(dna_input)
        if not cleaned:
            st.error("Lỗi: Chuỗi DNA không hợp lệ. Vui lòng nhập A, T, C, hoặc G.")
        else:
            result_df = infer_all_models(vectorizer, models, cleaned)

            if sort_mode == "Confidence":
                result_df = result_df.sort_values(by="Confidence", ascending=False)
            elif sort_mode == "Model name":
                result_df = result_df.sort_values(by="Model", ascending=True)

            top_row = result_df.iloc[0]
            avg_risk = float(result_df["Diabetic_Prob"].mean())
            high_risk_count = int((result_df["Diabetic_Prob"] >= threshold).sum())
            consensus = int((result_df["Prediction"] == top_row["Prediction"]).sum())

            # Metrics
            st.subheader("💡 Tóm tắt rủi ro (Summary)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Top Model",
                display_model_name(top_row["Model"]),
                str(top_row["Prediction"]),
                delta_color="off"
            )
            m2.metric("Trung bình rủi ro", f"{avg_risk*100:.1f}%", f"{high_risk_count}/{len(result_df)} model > {threshold*100:.0f}%", delta_color="inverse")
            m3.metric("Độ tương đồng (Consensus)", f"{consensus}/{len(result_df)}", f"Đồng thuận: {top_row['Prediction']}", delta_color="off")
            m4.metric("Dự đoán top 1", f"{top_row['Confidence']:.1f}% Tin cậy", top_row["Prediction"], delta_color="inverse" if top_row["Prediction"] == "Diabetic" else "normal")

            decision = compute_final_decision(result_df, threshold)
            st.session_state["last_result_df"] = result_df.copy()
            st.session_state["last_cleaned_seq"] = cleaned
            st.session_state["last_decision"] = decision

            st.write("---")
            render_final_decision_panel(decision, threshold)

            st.write("---")
            
            # Use two columns for Chart and Table
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.subheader("📊 Xu hướng Dự đoán rủi ro")
                trend_df = result_df.copy().sort_values(by="Diabetic_Prob", ascending=False)
                trend_df["Diabetic_Prob"] = trend_df["Diabetic_Prob"] * 100
                # Using a bar chart as it looks cleaner for comparisons
                fig = px.bar(
                    trend_df,
                    x="Model",
                    y="Diabetic_Prob",
                    color="Prediction",
                    title="Khả năng Đái tháo đường (%) trên từng Mô hình",
                    color_discrete_map={"Diabetic": "#ff4b4b", "Non-diabetic": "#00ea00"},
                    height=400
                )
                fig.update_layout(yaxis_title="Xác suất (%)", xaxis_title="", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("📋 Bảng chi tiết")
                
                display_df = result_df.copy()
                display_df["Model"] = display_df["Model"].apply(display_model_name)
                display_df["Risk_Level"] = display_df["Diabetic_Prob"].apply(lambda prob: risk_label(prob, threshold))
                display_df["Diabetic_Prob"] = (display_df["Diabetic_Prob"] * 100).round(2)
                display_df["Confidence"] = display_df["Confidence"].round(2)
                
                styled_df = display_df[["Model", "Diabetic_Prob", "Prediction", "Risk_Level"]].style.apply(risk_row_style, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=400)

            with st.expander("🔍 Xem thêm chi tiết thẻ điểm từng mô hình (Score Cards)"):
                cols = st.columns(4)
                for idx, row in result_df.reset_index(drop=True).iterrows():
                    with cols[idx % 4]:
                        render_score_card(row["Model"], row)


with tab_explain:
    st.subheader("🔍 Explainability & Reliability")

    if st.session_state.get("last_result_df") is None or st.session_state.get("last_cleaned_seq") is None:
        st.info("Hãy chạy dự đoán ở tab Predict trước, sau đó quay lại tab này để xem giải thích.")
    else:
        result_df = st.session_state["last_result_df"]
        cleaned_seq = st.session_state["last_cleaned_seq"]
        decision = st.session_state["last_decision"]

        render_reliability_analysis(result_df, decision)
        st.write("---")
        render_kmer_explanation(vectorizer, models, cleaned_seq, top_n=15)
        st.write("---")
        render_deep_saliency(models, cleaned_seq)


with tab_leaderboard:
    st.subheader("🏆 Training Leaderboard")
    leaderboard = load_benchmark_table()
    if leaderboard is None:
        st.info("Chưa tìm thấy benchmark. Hãy chạy `python DNA_Cls.py`")
    else:
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        
        st.subheader("Trực quan so sánh")
        long_df = leaderboard.melt(id_vars=["model"], value_vars=["accuracy", "f1", "roc_auc"], var_name="Metric", value_name="Score")
        fig_metrics = px.bar(long_df, x="model", y="Score", color="Metric", barmode="group", height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)

with tab_guidance:
    st.markdown("- **Classical ML (XGBoost/NuSVC):** Hoạt động nhanh, cực kì ổn định, cung cấp pipeline so sánh cơ bản.")
    st.markdown("- **Deep Learning (DeepTextCNN):** Tự động nhận diện cấu trúc token dài, khả năng nhận dạng k-mer linh động.")
    st.markdown("🎯 **Khuyến nghị:** Dùng Ensemble/Consensus của hệ thống để đưa ra chẩn đoán phòng ngừa chính xác nhất.")