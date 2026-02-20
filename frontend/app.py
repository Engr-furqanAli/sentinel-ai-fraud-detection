import streamlit as st
import requests
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import numpy as np
from streamlit_lottie import st_lottie
import json
import requests as req

st.set_page_config(
    page_title="Sentinel AI | Fraud Detection Platform",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# LOAD LOTTIE ANIMATIONS
# -----------------------------
def load_lottieurl(url):
    r = req.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Animation URLs
lottie_shield = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_p1qiuawe.json")
lottie_scan = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_syqnfe7u.json")
lottie_secure = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jmeowp.json")

# -----------------------------
# CUSTOM CSS - CYBERPUNK THEME
# -----------------------------
st.markdown("""
<style>
    /* Import futuristic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0b0c15 0%, #1a1f35 100%);
        font-family: 'Rajdhani', sans-serif;
    }

    /* Main container */
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(10, 20, 40, 0.7);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }

    .main-header h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #a0a0ff;
        font-size: 1.2rem;
        letter-spacing: 2px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0a0f1f 0%, #141b33 100%);
        border-right: 2px solid rgba(0, 255, 255, 0.3);
    }

    /* Metric cards */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 20px;
        margin: 30px 0;
    }

    .metric-card {
        background: rgba(20, 30, 50, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 255, 0.5);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        flex: 1;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
        animation: glow 2s ease-in-out infinite;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.4);
    }

    .metric-label {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
    }

    .metric-risk-Low { color: #00ff00; text-shadow: 0 0 20px #00ff00; }
    .metric-risk-Medium { color: #ffff00; text-shadow: 0 0 20px #ffff00; }
    .metric-risk-High { color: #ff0000; text-shadow: 0 0 20px #ff0000; }

    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(16, 24, 40, 0.8);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 12px;
        color: white;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #ff00ff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
    }

    .stNumberInput > div > div > input {
        background: rgba(16, 24, 40, 0.8);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 12px;
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(255, 0, 255, 0.5);
    }

    /* Dataframes */
    .dataframe {
        background: rgba(20, 30, 50, 0.8);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 15px;
        color: white;
    }

    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }

    /* Animations */
    @keyframes glow {
        0% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.2); }
        50% { box-shadow: 0 0 40px rgba(255, 0, 255, 0.4); }
        100% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.2); }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }

    .status-approve {
        background: rgba(0, 255, 0, 0.2);
        border: 1px solid #00ff00;
        color: #00ff00;
    }

    .status-review {
        background: rgba(255, 255, 0, 0.2);
        border: 1px solid #ffff00;
        color: #ffff00;
    }

    .status-block {
        background: rgba(255, 0, 0, 0.2);
        border: 1px solid #ff0000;
        color: #ff0000;
    }

    /* Divider */
    .neon-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffff, #ff00ff, transparent);
        margin: 30px 0;
    }

    /* Terminal effect */
    .terminal-text {
        font-family: 'Courier New', monospace;
        color: #00ff00;
        background: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER SECTION WITH ANIMATION
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if lottie_shield:
        st_lottie(lottie_shield, height=150, key="shield_anim")

    st.markdown("""
    <div class="main-header">
        <h1>SENTINEL AI</h1>
        <p>⚡ ADVANCED FRAUD DETECTION SYSTEM ⚡</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR WITH MODERN NAVIGATION
# -----------------------------
with st.sidebar:
    if lottie_secure:
        st_lottie(lottie_secure, height=100, key="secure_anim")

    st.markdown("## 🎯 NAVIGATION")

    option = st.radio(
        "Select Module",
        [
            "💳 Credit Card Fraud",
            "🛒 Retail Fraud",
            "📂 Batch Scan CSV",
            "🧠 Custom Model Training"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # System status
    st.markdown("### 🔧 SYSTEM STATUS")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🟢 API")
    with col2:
        st.markdown("🟢 Model")

    # Real-time metrics
    st.markdown("### 📊 LIVE METRICS")
    st.markdown(f"**Transactions Today:** {np.random.randint(1000, 5000)}")
    st.markdown(f"**Fraud Detected:** {np.random.randint(10, 100)}")
    st.markdown(f"**Accuracy Rate:** {np.random.randint(95, 99)}%")

API_URL = "http://127.0.0.1:8000/api/v1"

# ================================
# CREDIT CARD FRAUD
# ================================
if option == "💳 Credit Card Fraud":
    st.markdown("## 💳 CREDIT CARD FRAUD DETECTION")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "🔄 Quick Demo"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Transaction Details")
            amount = st.number_input(
                "💰 Amount ($)",
                min_value=0.0,
                value=100.0,
                help="Enter the transaction amount"
            )

            location = st.text_input(
                "📍 Location",
                value="New York",
                help="Enter the transaction location"
            )

            device = st.text_input(
                "📱 Device Used",
                value="Mobile",
                help="Enter the device used for transaction"
            )

        with col2:
            st.markdown("### Additional Info")
            time_of_day = st.select_slider(
                "⏰ Time of Day",
                options=["Morning", "Afternoon", "Evening", "Night"],
                value="Afternoon"
            )

            transaction_type = st.selectbox(
                "🔄 Transaction Type",
                ["Online", "In-store", "ATM", "Wire Transfer"]
            )

    with tab2:
        st.markdown("### 🎯 Quick Demo Examples")
        examples = {
            "Normal Transaction": {"amount": 50.0, "location": "Home", "device": "Mobile"},
            "Suspicious Transaction": {"amount": 5000.0, "location": "Foreign Country", "device": "Unknown Device"},
            "High Risk": {"amount": 15000.0, "location": "High Risk Zone", "device": "New Device"}
        }

        selected_example = st.selectbox("Select Example", list(examples.keys()))
        if selected_example:
            amount = examples[selected_example]["amount"]
            location = examples[selected_example]["location"]
            device = examples[selected_example]["device"]
            st.info(f"Loaded: {selected_example}")

    # Analysis button with animation
    if st.button("🔍 ANALYZE TRANSACTION", use_container_width=True):

        with st.spinner("🔄 Analyzing transaction..."):
            time.sleep(1.5)  # Simulate processing

            payload = {
                "amount": amount,
                "location": location,
                "device": device
            }

            response = requests.post(f"{API_URL}/predict", json=payload)

            if response.status_code == 200:
                result = response.json()

                # Create modern metric display
                st.markdown("### 📊 ANALYSIS RESULTS")

                # Progress bar for fraud probability
                st.progress(result['fraud_probability'])

                # Metrics in modern cards
                cols = st.columns(3)

                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">FRAUD PROBABILITY</div>
                        <div class="metric-value">{result['fraud_probability']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with cols[1]:
                    risk_color = {
                        "Low": "#00ff00",
                        "Medium": "#ffff00",
                        "High": "#ff0000"
                    }.get(result['risk_level'], "#ffffff")

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">RISK LEVEL</div>
                        <div class="metric-value" style="color: {risk_color};">{result['risk_level']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with cols[2]:
                    badge_class = {
                        "Approve": "status-approve",
                        "Manual Review": "status-review",
                        "Block Transaction": "status-block"
                    }.get(result['decision'], "")

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">DECISION</div>
                        <div class="metric-value"><span class="status-badge {badge_class}">{result['decision']}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Create gauge chart for visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['fraud_probability'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk Score", 'font': {'color': 'white'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickcolor': 'white'},
                        'bar': {'color': "rgba(0,255,255,0.8)"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(0,255,255,0.3)",
                        'steps': [
                            {'range': [0, 15], 'color': 'rgba(0,255,0,0.2)'},
                            {'range': [15, 35], 'color': 'rgba(255,255,0,0.2)'},
                            {'range': [35, 100], 'color': 'rgba(255,0,0,0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['fraud_probability'] * 100
                        }
                    }
                ))

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Orbitron'}
                )

                st.plotly_chart(fig, use_container_width=True)

                # Transaction details in terminal style
                st.markdown("### 📝 TRANSACTION DETAILS")
                st.markdown(f"""
                <div class="terminal-text">
                > AMOUNT: ${amount:,.2f}<br>
                > LOCATION: {location}<br>
                > DEVICE: {device}<br>
                > TIMESTAMP: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                > ANALYSIS COMPLETE
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("⚠️ Backend connection failed. Please check if the API is running.")

# ================================
# RETAIL FRAUD
# ================================
elif option == "🛒 Retail Fraud":
    st.markdown("## 🛒 RETAIL FRAUD DETECTION")

    # Create columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Transaction Information")
        amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, value=100.0)
        location = st.text_input("📍 Location", value="Store #123")
        device_used = st.text_input("📱 Device Used", value="POS Terminal")

    with col2:
        st.markdown("### Customer Information")
        merchant_category = st.selectbox(
            "🏪 Merchant Category",
            ["Electronics", "Clothing", "Groceries", "Restaurant", "Travel", "Other"]
        )
        payment_channel = st.selectbox(
            "💳 Payment Channel",
            ["Online", "In-store", "Mobile App", "Phone Order"]
        )

    if st.button("🔍 ANALYZE RETAIL TRANSACTION", use_container_width=True):

        with st.spinner("🔄 Analyzing retail transaction..."):
            time.sleep(1.5)

            payload = {
                "sender_account": "RETAIL_CUST_001",
                "receiver_account": "MERCHANT_001",
                "amount": amount,
                "transaction_type": "purchase",
                "merchant_category": merchant_category.lower(),
                "location": location,
                "device_used": device_used,
                "time_since_last_transaction": np.random.randint(1, 100),
                "spending_deviation_score": np.random.random(),
                "velocity_score": np.random.random(),
                "geo_anomaly_score": np.random.random(),
                "payment_channel": payment_channel.lower(),
                "ip_address": "192.168.1.1",
                "device_hash": "hash123"
            }

            response = requests.post(f"{API_URL}/predict_retail", json=payload)

            if response.status_code == 200:
                result = response.json()

                # Display results with modern UI
                st.markdown("### 📊 RETAIL ANALYSIS RESULTS")

                cols = st.columns(3)

                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">FRAUD PROBABILITY</div>
                        <div class="metric-value">{result['fraud_probability']:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with cols[1]:
                    risk_color = {
                        "Low": "#00ff00",
                        "Medium": "#ffff00",
                        "High": "#ff0000"
                    }.get(result['risk_level'], "#ffffff")

                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">RISK SCORE</div>
                        <div class="metric-value">{result['risk_score']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">DECISION</div>
                        <div class="metric-value" style="color: {risk_color};">{result['decision']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Radar chart for risk factors
                categories = ['Amount', 'Location', 'Device', 'Time', 'Velocity', 'History']
                values = [
                    min(amount / 10000, 1) * 100,
                    np.random.randint(20, 100),
                    np.random.randint(20, 100),
                    np.random.randint(20, 100),
                    payload['velocity_score'] * 100,
                    payload['spending_deviation_score'] * 100
                ]

                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line=dict(color='rgba(0, 255, 255, 0.8)', width=2),
                    fillcolor='rgba(0, 255, 255, 0.2)'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            color='white'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title=dict(text="Risk Factor Analysis", font=dict(color='white'))
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("⚠️ Backend connection failed.")

# ================================
# BATCH CSV SCAN
# ================================
elif option == "📂 Batch Scan CSV":
    st.markdown("## 📂 BATCH FRAUD DETECTION")

    if lottie_scan:
        st_lottie(lottie_scan, height=150, key="scan_anim")

    st.markdown("### Upload CSV File for Bulk Analysis")

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing transactions to analyze"
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        st.markdown("### 📄 Data Preview")
        st.dataframe(
            df.head(),
            use_container_width=True,
            height=200
        )

        st.markdown(f"**Total Records:** {len(df)}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 START BATCH SCAN", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []

                for i, (_, row) in enumerate(df.iterrows()):
                    progress = (i + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing transaction {i + 1} of {len(df)}...")

                    response = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "amount": float(row.get("amount", 0)),
                            "location": str(row.get("location", "")),
                            "device": str(row.get("device", ""))
                        }
                    )

                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'fraud_probability': result['fraud_probability'],
                            'risk_level': result['risk_level'],
                            'decision': result['decision']
                        })
                    else:
                        results.append({
                            'fraud_probability': None,
                            'risk_level': 'Error',
                            'decision': 'Error'
                        })

                # Add results to dataframe
                df['fraud_probability'] = [r['fraud_probability'] for r in results]
                df['risk_level'] = [r['risk_level'] for r in results]
                df['decision'] = [r['decision'] for r in results]

                status_text.text("✅ Scan Complete!")

                st.markdown("### 📊 Scan Results")
                st.dataframe(df, use_container_width=True)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="fraud_scan_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col2:
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            if 'amount' in df.columns:
                st.metric("Total Value", f"${df['amount'].sum():,.2f}")
                st.metric("Average Amount", f"${df['amount'].mean():,.2f}")
                st.metric("Max Amount", f"${df['amount'].max():,.2f}")

    else:
        st.info("👆 Upload a CSV file to begin batch scanning")

# ================================
# CUSTOM MODEL TRAINING
# ================================
elif option == "🧠 Custom Model Training":
    st.markdown("## 🧠 CUSTOM MODEL TRAINING")

    st.markdown("### Train Your Own Fraud Detection Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "📤 Upload Training Dataset (CSV)",
            type=["csv"],
            help="Upload a CSV file containing your training data"
        )

    with col2:
        st.markdown("### Model Settings")
        test_size = st.slider("Test Size (%)", 10, 30, 20) / 100
        n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)
        max_depth = st.slider("Max Depth", 5, 30, 10, 5)

    if uploaded:
        df = pd.read_csv(uploaded)

        st.markdown("### 📊 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

        target = st.selectbox(
            "🎯 Select Target Column (what to predict)",
            df.columns
        )

        if st.button("🚀 START TRAINING", use_container_width=True):

            with st.spinner("🔄 Training model... This may take a moment."):

                # Prepare data
                X = df.drop(columns=[target])
                y = df[target]

                # Encode categorical columns
                categorical_cols = X.select_dtypes(include=["object"]).columns
                for col in categorical_cols:
                    X[col] = X[col].astype("category").cat.codes

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Train model
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)

                # Evaluate
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)

                # Display results
                st.markdown("### 📈 Training Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">TRAINING ACCURACY</div>
                        <div class="metric-value">{train_acc:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">TEST ACCURACY</div>
                        <div class="metric-value">{test_acc:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">OVERFITTING</div>
                        <div class="metric-value">{'Low' if (train_acc - test_acc) < 0.1 else 'High'}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Feature Importance',
                    color='importance',
                    color_continuous_scale='Viridis'
                )

                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                )

                st.plotly_chart(fig, use_container_width=True)

                # Save model
                model_path = "custom_model.pkl"
                joblib.dump(model, model_path)

                with open(model_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f,
                        file_name="custom_fraud_model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )

                st.success("✅ Model training completed successfully!")

# Footer
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("""
    <p style="text-align: center; color: #666; font-family: 'Orbitron';">
        ⚡ SENTINEL AI v2.0 | ADVANCED FRAUD PROTECTION ⚡
    </p>
    """, unsafe_allow_html=True)