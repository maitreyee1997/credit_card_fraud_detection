"""
💳 Credit Card Fraud Detection - Upgraded Streamlit App
Streamlit v1.12 Compatible
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.model_training import load_model

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    /* Dark professional theme */
    .main { background-color: #0a0e1a; }

    .main-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0d1b2a 100%);
        border: 1px solid #00d4aa;
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 30px rgba(0, 212, 170, 0.2);
    }
    .main-header h1 {
        font-size: 2.2rem; margin: 0;
        color: #00d4aa;
        text-shadow: 0 0 20px rgba(0,212,170,0.5);
    }
    .main-header p { font-size: 0.95rem; margin: 0.4rem 0 0 0; color: #8892b0; }

    /* Result cards */
    .result-fraud {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; padding: 1.5rem; border-radius: 15px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(231,76,60,0.5);
        animation: pulse 2s infinite;
    }
    .result-legit {
        background: linear-gradient(135deg, #00b894, #00d4aa);
        color: white; padding: 1.5rem; border-radius: 15px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,212,170,0.4);
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem; font-weight: bold; color: #00d4aa;
        border-left: 4px solid #00d4aa; padding-left: 0.8rem;
        margin: 1rem 0 0.5rem 0;
    }

    /* Info box */
    .info-box {
        background: #1a1f35; padding: 1rem; border-radius: 10px;
        border: 1px solid #00d4aa33; margin: 0.5rem 0;
        font-size: 0.9rem; color: #ccd6f6;
    }

    /* Feature input section */
    .feature-section {
        background: #1a1f35; padding: 1rem; border-radius: 10px;
        border: 1px solid #2a3050; margin: 0.5rem 0;
    }

    /* Stats card */
    .stat-card {
        background: #1a1f35; padding: 1rem; border-radius: 10px;
        text-align: center; border: 1px solid #2a3050;
    }
    .stat-card .val { font-size: 1.5rem; font-weight: bold; color: #00d4aa; }
    .stat-card .lbl { font-size: 0.8rem; color: #8892b0; margin-top: 0.2rem; }

    /* Alert box */
    .alert-fraud {
        background: #2d1515; border: 1px solid #e74c3c;
        border-radius: 10px; padding: 0.8rem 1rem;
        color: #ff6b6b; margin: 0.5rem 0;
    }
    .alert-legit {
        background: #0d2b25; border: 1px solid #00d4aa;
        border-radius: 10px; padding: 0.8rem 1rem;
        color: #00d4aa; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ================================
# LOAD MODEL
# ================================
@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("model/fraud_detection_model.pkl")


# ================================
# PRESET TRANSACTIONS
# ================================
def get_preset(preset_type, feature_names):
    """Demo transactions for quick testing"""
    base = {f: 0.0 for f in feature_names}
    if preset_type == "fraud":
        # High risk values
        presets = {'Time': 406.0, 'Amount': 2125.87,
                   'V1': -3.04, 'V2': 3.98, 'V3': -4.21,
                   'V4': 4.46, 'V5': -3.08, 'V6': -1.33,
                   'V7': -4.35, 'V14': -9.17, 'V17': -5.83}
    else:
        # Normal transaction
        presets = {'Time': 52000.0, 'Amount': 45.20,
                   'V1': 1.19, 'V2': 0.26, 'V3': 0.16,
                   'V4': 0.45, 'V5': 0.23, 'V14': 0.15}
    base.update({k: v for k, v in presets.items() if k in base})
    return base


# ================================
# MAIN APP
# ================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💳 Credit Card Fraud Detection System</h1>
        <p>Real-Time ML-Powered Transaction Security | Random Forest | AUC: 0.975</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model = get_model()
    feature_names = list(model.feature_names_in_)

    # ---- TOP STATS ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-card"><div class="val">0.975</div><div class="lbl">ROC-AUC Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="val">94%</div><div class="lbl">Fraud Precision</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><div class="val">83%</div><div class="lbl">Fraud Recall</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="val">0.3</div><div class="lbl">Optimized Threshold</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- QUICK PRESETS ----
    st.markdown('<div class="section-title">⚡ Quick Test (Demo Transactions)</div>', unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        load_fraud = st.button("🔴 Load Fraud Transaction")
    with col_p2:
        load_legit = st.button("🟢 Load Legitimate Transaction")
    with col_p3:
        reset_btn = st.button("🔄 Reset All Fields")

    # Session state for input values
    if 'input_vals' not in st.session_state:
        st.session_state.input_vals = {f: 0.0 for f in feature_names}

    if load_fraud:
        st.session_state.input_vals = get_preset("fraud", feature_names)
        st.info("🔴 Fraud transaction loaded — scroll down and click Predict!")
    elif load_legit:
        st.session_state.input_vals = get_preset("legit", feature_names)
        st.info("🟢 Legitimate transaction loaded — scroll down and click Predict!")
    elif reset_btn:
        st.session_state.input_vals = {f: 0.0 for f in feature_names}

    # ---- INPUT + RESULT LAYOUT ----
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-title">📝 Transaction Features</div>', unsafe_allow_html=True)

        # Time & Amount first (most important)
        st.markdown("**🕐 Time & Amount**")
        r1, r2 = st.columns(2)
        user_input = {}

        with r1:
            if 'Time' in feature_names:
                user_input['Time'] = st.number_input(
                    "Time (seconds)", value=float(st.session_state.input_vals.get('Time', 0.0)),
                    help="Seconds elapsed since first transaction"
                )
        with r2:
            if 'Amount' in feature_names:
                user_input['Amount'] = st.number_input(
                    "Amount ($)", value=float(st.session_state.input_vals.get('Amount', 0.0)),
                    help="Transaction amount in dollars"
                )

        # V1-V14 features
        st.markdown("**🔢 PCA Features (V1–V14)**")
        v_features_1 = [f for f in feature_names if f.startswith('V') and int(f[1:]) <= 14]
        cols = st.columns(4)
        for i, feat in enumerate(v_features_1):
            with cols[i % 4]:
                user_input[feat] = st.number_input(
                    feat,
                    value=float(st.session_state.input_vals.get(feat, 0.0)),
                    format="%.4f",
                    key=f"input_{feat}"
                )

        # V15-V28 features
        st.markdown("**🔢 PCA Features (V15–V28)**")
        v_features_2 = [f for f in feature_names if f.startswith('V') and int(f[1:]) > 14]
        cols2 = st.columns(4)
        for i, feat in enumerate(v_features_2):
            with cols2[i % 4]:
                user_input[feat] = st.number_input(
                    feat,
                    value=float(st.session_state.input_vals.get(feat, 0.0)),
                    format="%.4f",
                    key=f"input_{feat}"
                )

        # Fill remaining features
        for feat in feature_names:
            if feat not in user_input:
                user_input[feat] = st.session_state.input_vals.get(feat, 0.0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 ANALYZE TRANSACTION")

    # ---- RIGHT COLUMN ----
    with col_right:
        st.markdown('<div class="section-title">🎯 Detection Result</div>', unsafe_allow_html=True)

        if predict_btn:
            input_df = pd.DataFrame([user_input], columns=feature_names)
            proba = model.predict_proba(input_df)[0][1]
            prob_pct = round(proba * 100, 2)
            is_fraud = proba >= 0.3

            # Gauge Chart
            gauge_color = "#e74c3c" if is_fraud else "#00d4aa"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability (%)",
                       'font': {'size': 15, 'color': '#ccd6f6'}},
                delta={'reference': 30,
                       'increasing': {'color': "#e74c3c"},
                       'decreasing': {'color': "#00d4aa"}},
                number={'font': {'color': gauge_color, 'size': 40}},
                gauge={
                    'axis': {'range': [0, 100],
                             'tickcolor': '#8892b0',
                             'tickfont': {'color': '#8892b0'}},
                    'bar': {'color': gauge_color},
                    'bgcolor': '#1a1f35',
                    'borderwidth': 1,
                    'bordercolor': '#2a3050',
                    'steps': [
                        {'range': [0, 30],  'color': '#0d2b25'},
                        {'range': [30, 60], 'color': '#2b2215'},
                        {'range': [60, 100],'color': '#2d1515'}
                    ],
                    'threshold': {
                        'line': {'color': "#e74c3c", 'width': 3},
                        'thickness': 0.75, 'value': 30
                    }
                }
            ))
            fig_gauge.update_layout(
                height=270,
                margin=dict(t=50, b=10, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ccd6f6'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Result card
            st.markdown("<br>", unsafe_allow_html=True)
            if is_fraud:
                st.markdown(f"""
                <div class="result-fraud">
                    🚨 FRAUD DETECTED!<br>
                    <small>Probability: {prob_pct}%</small>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="alert-fraud">
                    ⚠️ <b>Immediate Actions:</b><br>
                    • Transaction block karo turant<br>
                    • Customer ko alert SMS bhejo<br>
                    • Fraud team ko notify karo<br>
                    • Card temporarily freeze karo
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    ✅ LEGITIMATE TRANSACTION<br>
                    <small>Probability: {prob_pct}%</small>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="alert-legit">
                    ✅ <b>Transaction Approved</b><br>
                    • Normal transaction pattern<br>
                    • No suspicious activity detected<br>
                    • Safe to process
                </div>
                """, unsafe_allow_html=True)

            # Probability bar
            st.markdown('<div class="section-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=['Legitimate', 'Fraud'],
                y=[round(100 - prob_pct, 2), prob_pct],
                marker_color=['#00d4aa', '#e74c3c'],
                text=[f'{100-prob_pct:.1f}%', f'{prob_pct:.1f}%'],
                textposition='outside',
                textfont={'size': 13, 'color': '#ccd6f6'}
            ))
            fig_bar.update_layout(
                height=220,
                margin=dict(t=20, b=10, l=10, r=10),
                yaxis_range=[0, 120],
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#ccd6f6'},
                xaxis={'tickfont': {'color': '#ccd6f6'}},
                yaxis={'tickfont': {'color': '#8892b0'}}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <h4 style="color:#00d4aa">ℹ️ Kaise use karein?</h4>
                <ol>
                    <li>Quick test ke liye upar <b>Load Fraud/Legit</b> button dabao</li>
                    <li>Ya manually features bharein</li>
                    <li><b>ANALYZE TRANSACTION</b> dabao</li>
                    <li>Result yahan dikhega</li>
                </ol>
                <br>
                <b style="color:#00d4aa">Threshold = 30%</b><br>
                <span style="color:#8892b0">Fraud recall maximize karne ke liye optimized</span>
            </div>
            """, unsafe_allow_html=True)

    # ---- MODEL PERFORMANCE SECTION ----
    st.markdown("---")
    st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Threshold Comparison", "🎯 Business Impact"])

    with tab1:
        # Threshold comparison chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        default_scores  = [99.9, 97.0, 74.0, 84.0]
        optimized_scores = [99.8, 94.0, 83.0, 88.0]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Default (0.5)',
            x=metrics, y=default_scores,
            marker_color='#3498db',
            text=[f'{v}%' for v in default_scores],
            textposition='outside'
        ))
        fig_comp.add_trace(go.Bar(
            name='Optimized (0.3) ✅',
            x=metrics, y=optimized_scores,
            marker_color='#00d4aa',
            text=[f'{v}%' for v in optimized_scores],
            textposition='outside'
        ))
        fig_comp.update_layout(
            barmode='group',
            height=320,
            yaxis_range=[60, 105],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ccd6f6'},
            legend={'font': {'color': '#ccd6f6'}},
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        📌 <b style="color:#00d4aa">Threshold 0.3</b> se Recall 74% → 83% ho gaya!
        Matlab 9% zyada fraud transactions pakde jaate hain.
        F1-Score bhi improve hua: 84% → 88% ✅
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        # Business impact
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("""
            <div class="info-box">
                <h4 style="color:#e74c3c">❌ Default Threshold (0.5)</h4>
                <ul>
                    <li>Recall: 74%</li>
                    <li>26% fraud miss ho jaata hai</li>
                    <li>Per 100 fraud = 26 undetected</li>
                    <li>Avg fraud = $2000 → Loss = <b>$52,000</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col_b2:
            st.markdown("""
            <div class="info-box">
                <h4 style="color:#00d4aa">✅ Optimized Threshold (0.3)</h4>
                <ul>
                    <li>Recall: 83%</li>
                    <li>Only 17% fraud miss hota hai</li>
                    <li>Per 100 fraud = 17 undetected</li>
                    <li>Avg fraud = $2000 → Loss = <b>$34,000</b></li>
                </ul>
                <b style="color:#00d4aa">💰 $18,000 savings per 100 fraud cases!</b>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🧠 Model Details (Click to expand)"):
        st.markdown("""
        | Detail | Value |
        |--------|-------|
        | Algorithm | Random Forest Classifier |
        | n_estimators | 100 |
        | max_depth | 10 |
        | min_samples_split | 5 |
        | class_weight | balanced |
        | Tuning | RandomizedSearchCV (3-fold) |
        | Dataset | 284,807 transactions |
        | Fraud cases | 492 (0.17%) |
        | ROC-AUC | 0.975 |
        """)

    st.markdown("""
    ---
    <center>
    <small style="color:#8892b0">
    💳 Credit Card Fraud Detection System |
    Random Forest + Threshold Optimization |
    Dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" style="color:#00d4aa">Kaggle Credit Card Fraud</a> |
    ⚠️ For educational & business decision support only
    </small>
    </center>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()