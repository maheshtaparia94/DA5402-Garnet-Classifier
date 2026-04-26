import io
import os
import uuid
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from prometheus_client import Counter, Gauge, start_http_server

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Garnet Raman Classifier",
    page_icon="💎",
    layout="wide")

#  Prometheus metrics
@st.cache_resource
def init_metrics():
    try:
        start_http_server(8002)
    except OSError:
        pass
    requests_total = Counter("garnet_webapp_requests", "Requests per session", ["session_id", "mode"])
    total_sessions = Gauge("garnet_total_sessions", "Total unique browser sessions opened")
    errors_total = Counter("garnet_webapp_errors", "Webapp errors", ["error_type"])
    return requests_total, total_sessions, errors_total

REQUESTS_TOTAL, TOTAL_SESSIONS, ERRORS_TOTAL = init_metrics()

# Session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

if "session_registered" not in st.session_state:
    st.session_state["session_registered"] = True
    TOTAL_SESSIONS.inc()   # increment once per new browser session

if "feedback_submitted" not in st.session_state:
    st.session_state["feedback_submitted"] = {}

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = None

session_id = st.session_state["session_id"]


# API helpers
def check_api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False

def get_classes():
    try:
        return requests.get(f"{API_URL}/classes", timeout=3).json()["classes"]
    except Exception:
        return []

def predict(file_bytes, filename):
    r = requests.post(f"{API_URL}/predict",
                      files={"file": (filename, file_bytes)}, timeout=30)
    r.raise_for_status()
    return r.json()["predictions"]

def submit_feedback(filename, ground_truth):
    r = requests.post(f"{API_URL}/feedback",
                      json={"filename": filename, "ground_truth": ground_truth},
                      timeout=10)
    r.raise_for_status()
    return r.json()

def get_pending():
    r = requests.get(f"{API_URL}/predictions/pending", timeout=10)
    r.raise_for_status()
    return r.json()["predictions"]

def get_spectrum(prediction_id):
    r = requests.get(f"{API_URL}/predictions/{prediction_id}/spectrum", timeout=10)
    r.raise_for_status()
    return r.json()

def parse_spectrum(content):
    lines = content.decode("utf-8").strip().splitlines()
    wn, intensity = [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                wn.append(float(parts[0]))
                intensity.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(wn), np.array(intensity)

def spectrum_chart(wn, intensity, title="Spectrum", color="#00b4d8"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wn, y=intensity, mode="lines",
                             line=dict(color=color, width=1.5)))
    fig.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black"), margin=dict(l=40, r=20, t=40, b=40), height=280,
    )
    return fig


# Sidebar
with st.sidebar:
    st.title("💎 Garnet Classfication")
    st.caption("Raman Spectrum Classifier")
    st.divider()
    api_ok = check_api_health()
    if api_ok:
        st.success("API Online")
    else:
        st.error("API Offline")
    st.divider()
    st.caption(f"Session: `{session_id}`")

if not api_ok:
    st.error("API is offline. Please ensure the FastAPI service is running.")
    st.stop()

CLASSES = get_classes() or []


#  Tabs
tab_predict, tab_feedback, tab_pipeline, tab_help = st.tabs([
    "🔬 Predict", "📝 Pending Feedback", "📊 Pipeline", "❓ Help"])

with tab_predict:
    st.header("Upload Spectrum")
    uploaded = st.file_uploader("Upload .txt or .zip", type=["txt", "zip"],
                                help=".txt for single, .zip for multiple spectra")

    if not uploaded:
        st.info("Upload a spectrum file to get started.")
    else:
        file_bytes = uploaded.read()
        is_zip = uploaded.name.endswith(".zip")
        mode = "bulk" if is_zip else "single"

        if st.session_state["last_filename"] != uploaded.name:
            try:
                with st.spinner("Classifying..."):
                    predictions = predict(file_bytes, uploaded.name)
                st.session_state["last_prediction"] = predictions
                st.session_state["last_filename"]   = uploaded.name
                REQUESTS_TOTAL.labels(session_id=session_id, mode=mode).inc()
            except requests.exceptions.ConnectionError:
                ERRORS_TOTAL.labels(error_type="connection_error").inc()
                st.error("Cannot connect to API.")
                st.stop()
            except Exception as e:
                ERRORS_TOTAL.labels(error_type="unexpected").inc()
                st.error(f"Error: {e}")
                st.stop()
        else:
            predictions = st.session_state["last_prediction"]

        try:
            # Parse spectra for chart
            spectra = []
            if is_zip:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    for name in zf.namelist():
                        if name.endswith(".txt"):
                            wn, inten = parse_spectrum(zf.read(name))
                            spectra.append((name, wn, inten))
            else:
                wn, inten = parse_spectrum(file_bytes)
                spectra.append((uploaded.name, wn, inten))

            # Spectrum chart
            colors = ["#00b4d8","#f77f00","#06d6a0","#ef476f","#8338ec"]
            fig = go.Figure()
            for i, (name, wn, inten) in enumerate(spectra):
                fig.add_trace(go.Scatter(x=wn, y=inten, mode="lines",
                                         name=name, line=dict(color=colors[i % 5], width=1.5)))
            fig.update_layout(
                xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity",
                legend=dict(orientation="h", y=1.02),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="black"), margin=dict(l=40,r=20,t=40,b=40), height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

            # Single result
            if not is_zip:
                pred = predictions[0]
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🔬 Prediction")
                    st.metric("Class", pred["predicted_class"])
                    st.metric("Confidence", f"{pred['confidence']:.1%}")
                    if pred["confidence"] < 0.50:
                        st.warning("⚠️ Low confidence — saved for expert review")
                with col2:
                    st.subheader("📈 Probabilities")
                    probs  = pred["probabilities"]
                    pdf    = pd.DataFrame({"Class": list(probs.keys()),
                                           "Prob": list(probs.values())}).sort_values("Prob")
                    fig_b  = go.Figure(go.Bar(
                        x=pdf["Prob"], y=pdf["Class"], orientation="h",
                        marker_color=["#ef476f" if c == pred["predicted_class"] else "#00b4d8"
                                      for c in pdf["Class"]]))
                    fig_b.update_layout(xaxis=dict(range=[0,1], tickformat=".0%"),
                                        plot_bgcolor="white", paper_bgcolor="white",
                                        font=dict(color="black"),
                                        margin=dict(l=10,r=10,t=10,b=10), height=200)
                    st.plotly_chart(fig_b, use_container_width=True)

                st.divider()
                fname = pred["filename"]
                if st.session_state["feedback_submitted"].get(fname):
                    st.success("✅ Feedback already submitted")
                else:
                    st.subheader("📝 Submit Feedback")
                    class_options = list(pred["probabilities"].keys())
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        gt = st.selectbox("Correct class", class_options,
                                          index=class_options.index(pred["predicted_class"]),
                                          key=f"fb_{fname}")
                    with c2:
                        st.write("")
                        st.write("")
                        if st.button("Submit", key=f"btn_{fname}", type="primary"):
                            try:
                                result = submit_feedback(fname, gt)
                                st.session_state["feedback_submitted"][fname] = True
                                if result["is_wrong"]:
                                    st.error(f"❌ Wrong — correct: {gt}")
                                else:
                                    st.success("✅ Confirmed correct")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

            # Bulk results
            else:
                st.subheader(f"🔬 Results — {len(predictions)} spectra")
                for pred in predictions:
                    fname = pred["filename"]
                    conf  = pred["confidence"]
                    icon  = "🟢" if conf >= 0.85 else ("🟡" if conf >= 0.50 else "🔴")
                    with st.expander(f"{icon} {fname}  →  **{pred['predicted_class']}**  ({conf:.1%})"):
                        c1, c2, c3 = st.columns([2, 2, 2])
                        with c1:
                            st.metric("Class", pred["predicted_class"])
                            st.metric("Confidence", f"{conf:.1%}")
                            if conf < 0.50:
                                st.warning("⚠️ Low confidence")
                        with c2:
                            probs = pred["probabilities"]
                            st.dataframe(pd.DataFrame({"Class": list(probs.keys()),
                                                       "Prob": [f"{v:.1%}" for v in probs.values()]}),
                                         hide_index=True, use_container_width=True)
                        with c3:
                            if st.session_state["feedback_submitted"].get(fname):
                                st.success("✅ Submitted")
                            else:
                                class_opts = list(pred["probabilities"].keys())
                                gt = st.selectbox("Ground truth", class_opts,
                                                  index=class_opts.index(pred["predicted_class"]),
                                                  key=f"fb_{fname}", label_visibility="collapsed")
                                if st.button("Submit", key=f"btn_{fname}", type="primary"):
                                    try:
                                        result = submit_feedback(fname, gt)
                                        st.session_state["feedback_submitted"][fname] = True
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed: {e}")

        except requests.exceptions.ConnectionError:
            ERRORS_TOTAL.labels(error_type="connection_error").inc()
            st.error("Cannot connect to API.")
        except Exception as e:
            ERRORS_TOTAL.labels(error_type="unexpected").inc()
            st.error(f"Error: {e}")


# PENDING FEEDBACK
with tab_feedback:
    st.header("📝 Pending Feedback")
    st.caption("Predictions without ground truth. Submit feedback to improve the model.")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        pending = get_pending()
    except Exception as e:
        st.error(f"Could not load pending predictions: {e}")
        pending = []

    if not pending:
        st.success("✅ All predictions have feedback. Nothing pending.")
    else:
        st.info(f"**{len(pending)}** prediction(s) awaiting feedback.")
        st.divider()

        for pred in pending:
            pid   = pred["id"]
            fname = pred["filename"]
            conf  = pred["confidence"]
            drift = pred["drift_score"]

            # Flag indicators
            flags = []
            if pred["is_low_confidence"]:
                flags.append("🟡 Low Confidence")
            if pred["is_drifted"]:
                flags.append("🔴 High Drift")
            flag_str = "  ".join(flags) if flags else "🟢 Normal"

            with st.expander(
                f"**{fname}**  →  {pred['predicted_class']}  ({conf:.1%})  |  {flag_str}",
                expanded=True
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Show spectrum chart
                    try:
                        spec = get_spectrum(pid)
                        if spec["wavenumber"]:
                            fig = spectrum_chart(spec["wavenumber"], spec["intensity"],
                                                 title=fname)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.caption("Spectrum file not available")

                with col2:
                    st.metric("Predicted Class", pred["predicted_class"])
                    st.metric("Confidence",      f"{conf:.1%}")
                    st.metric("Drift Score",     f"{drift:.4f}")
                    st.caption(f"Recorded: {pred['timestamp'][:19]}")

                    if st.session_state["feedback_submitted"].get(f"pending_{pid}"):
                        st.success("✅ Submitted")
                    else:
                        gt = st.selectbox(
                            "Correct class",
                            CLASSES,
                            index=CLASSES.index(pred["predicted_class"]) if pred["predicted_class"] in CLASSES else 0,
                            key=f"pfb_{pid}"
                        )
                        if st.button("Submit Feedback", key=f"pbtn_{pid}", type="primary"):
                            try:
                                result = submit_feedback(fname, gt)
                                st.session_state["feedback_submitted"][f"pending_{pid}"] = True
                                if result["is_wrong"]:
                                    st.error(f"❌ Wrong → moved to data/labeled/")
                                else:
                                    st.success("✅ Confirmed correct")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

with tab_pipeline:
    st.header("ML Pipeline")

    # ── Tool Links ─────────────────────────────────────────────
    st.subheader("🔗 Pipeline Tools")
    c1, c2, c3, c4 = st.columns(4)
    c1.link_button("MLflow UI",  "http://localhost:5000")
    c2.link_button("Airflow UI", "http://localhost:8080")
    c3.link_button("Grafana",    "http://localhost:3000/d/garnet-classifier/garnet-raman-classifier-dashboard")
    c4.link_button("Prometheus", "http://localhost:9090")

    st.divider()

    # ── DVC Pipeline Stages ────────────────────────────────────
    st.subheader("🔄 DVC Pipeline Stages")
    st.graphviz_chart("""
        digraph {
            rankdir=TB
            node [shape=box style=filled fillcolor="#1e3a5f" fontcolor=white color="#00b4d8" fontname=monospace]
            edge [color="#00b4d8"]
            raw  [label="data/raw.dvc"]
            golden [label="data/golden.dvc"]
            raw -> preprocess
            preprocess -> split_data
            split_data -> baseline_stats
            split_data -> train
            preprocess -> eda
            split_data -> eda
            train -> evaluate_register
            golden -> evaluate_register
        }
    """, use_container_width=True)

    st.divider()

    # ── Recent MLflow Runs ─────────────────────────────────────
    st.subheader("📈 Recent Training Runs")
    runs = []
    try:
        resp = requests.get(f"{API_URL}/pipeline/runs", timeout=5)
        runs_data = resp.json().get("runs", [])
        if runs_data:
            df = pd.DataFrame(runs_data)
            st.dataframe(df,
                        use_container_width=True,
                        height=400,
                        hide_index=False)
        else:
            st.info("No runs found.")
    except Exception as e:
        st.warning(f"Cannot reach API: {e}")
    st.divider()

    # ── Airflow DAG Visualization + Status ────────────────────
    st.subheader("🌀 Airflow Pipeline")

    # Static DAG graph
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.graphviz_chart("""
            digraph {
                rankdir=LR
                size="6,2"
                node [shape=box style=filled fillcolor="#1e3a5f" fontcolor=white color="#00b4d8" fontname=monospace fontsize=12]
                edge [color="#00b4d8"]
                dvc_repro -> git_push
            }
        """, use_container_width=True)

    # DAG runs table + task status
    try:
        airflow_url = os.environ.get("AIRFLOW_URL", "http://localhost:8080")

        runs_resp = requests.get(
            f"{airflow_url}/api/v1/dags/garnet_pipeline/dagRuns",
            auth=("airflow", "airflow"),
            params={"limit": 10, "order_by": "-start_date"},
            timeout=5)

        if runs_resp.status_code == 200:
            dag_runs = runs_resp.json().get("dag_runs", [])
            if dag_runs:
                rows = []
                for r in dag_runs:
                    dur = "running"
                    if r.get("end_date") and r.get("start_date"):
                        dur = f"{int((pd.Timestamp(r['end_date']) - pd.Timestamp(r['start_date'])).total_seconds())}s"
                    rows.append({
                        "Run ID":   r["dag_run_id"][:25],
                        "Status":   r["state"].upper(),
                        "Started":  r["start_date"][:19] if r["start_date"] else "-",
                        "Duration": dur,
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df,
                            use_container_width=True,
                            height=(len(df) + 1) * 35 + 3)

                # Latest run task status
                st.write("**Latest Run — Task Status:**")
                latest_run_id = dag_runs[0]["dag_run_id"]
                task_resp = requests.get(
                    f"{airflow_url}/api/v1/dags/garnet_pipeline/dagRuns/{latest_run_id}/taskInstances",
                    auth=("airflow", "airflow"),
                    timeout=5
                )
                if task_resp.status_code == 200:
                    task_instances = task_resp.json().get("task_instances", [])
                    if task_instances:
                        cols = st.columns(len(task_instances))
                        for i, t in enumerate(task_instances):
                            state = t.get("state") or "none"
                            icon  = "🟢" if state == "success" \
                                    else "🔴" if state == "failed" \
                                    else "🟡" if state == "running" \
                                    else "⚪"
                            cols[i].metric(
                                t["task_id"].replace("_", " ").title(),
                                f"{icon} {state.upper()}"
                            )
            else:
                st.info("No DAG runs yet.")
        else:
            st.info("Airflow not reachable — visit http://localhost:8080")
    except Exception:
        st.info("Airflow not reachable — visit http://localhost:8080")


# ── Help Tab ───────────────────────────────────────────────────
with tab_help:
    st.header("User Manual")
    st.caption("Quick guide for using the Garnet Raman Classifier")

    st.subheader("🔬 What Is This App?")
    st.write(
        "This application classifies Raman spectra of garnet minerals into one of five types: "
        "Almandine, Andradite, Grossular, Pyrope, or Spessartine."
    )

    st.divider()

    st.subheader("📤 How To Predict")
    st.markdown("""
        1. Go to the **Predict** tab
        2. Upload a `.txt` spectrum file or a `.zip` of multiple spectra
        3. The file must contain two columns: **wavenumber** and **intensity**
        4. Click the file uploader — prediction runs automatically
        5. View the predicted class and confidence score
    """)

    st.divider()

    st.subheader("📊 Understanding Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", "≥ 70%")
        st.caption("High confidence — reliable prediction")
    with col2:
        st.metric("Confidence", "50–70%")
        st.caption("Low confidence — review recommended")
    with col3:
        st.metric("Drift Score", "> 0.3")
        st.caption("Spectrum differs from training data")

    st.divider()

    st.subheader("✅ How To Submit Feedback")
    st.markdown("""
        1. Go to the **Pending Feedback** tab
        2. Find predictions without ground truth
        3. Select the correct class from the dropdown
        4. Click **Submit Feedback**
        5. Wrong predictions are saved for model retraining
    """)

    st.divider()

    st.subheader("📁 Accepted File Format")
    st.code("""# Example spectrum.txt
        200.6    733.9
        201.5    493.0
        202.6    964.2
        # Two columns: wavenumber (cm⁻¹) and intensity
        # Tab or space separated""", language="text")

    st.divider()

    st.subheader("❓ FAQ")
    with st.expander("What file types are supported?"):
        st.write("Single `.txt` spectrum files or `.zip` archives containing multiple `.txt` files.")
    with st.expander("What if my prediction shows Low Confidence?"):
        st.write("The model is uncertain. Please verify visually and submit correct feedback.")
    with st.expander("What does Drift Score mean?"):
        st.write("A high drift score means your spectrum differs significantly from training data. Results may be less reliable.")
    with st.expander("How do I improve model accuracy?"):
        st.write("Submit feedback with correct labels. Wrong predictions are automatically collected for model retraining.")