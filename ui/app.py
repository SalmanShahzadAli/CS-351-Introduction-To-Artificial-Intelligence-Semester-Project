# Fraud Detection System - Streamlit UI
# File: ui/app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR DARK THEME
# ============================================

st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    /* Sub Headers */
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4ff;
        margin-top: 20px;
        margin-bottom: 10px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 30, 50, 0.8);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #7c3aed;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
    }
    
    /* Fraud Alert - Enhanced */
    .fraud-alert {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(185, 28, 28, 0.3));
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
        backdrop-filter: blur(10px);
    }
    
    /* Safe Alert - Enhanced */
    .safe-alert {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.3));
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #22c55e;
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.4);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #7c3aed;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: rgba(30, 30, 50, 0.6);
        color: #e0e0e0;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.8);
        transform: translateY(-2px);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: rgba(30, 30, 50, 0.6) !important;
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(30, 30, 50, 0.6);
        color: #00d4ff;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
    
    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e0e0e0;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(30, 30, 50, 0.8);
        border-radius: 10px;
        border-left: 4px solid #7c3aed;
    }
    
    /* Divider */
    hr {
        border-color: #7c3aed;
        opacity: 0.3;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND DATA
# ============================================
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        with open('../models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
@st.cache_resource
def load_all_models():
    """Load all trained models for ensemble"""
    models = {}
    try:
        with open('../models/logistic_regression_baseline.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
        with open('../models/random_forest_model.pkl', 'rb') as f:
            models['rf'] = pickle.load(f)
        with open('../models/svm_model.pkl', 'rb') as f:
            models['svm'] = pickle.load(f)
        with open('../models/xgboost_model.pkl', 'rb') as f:
            models['xgb'] = pickle.load(f)
        return models
    except Exception as e:
        st.warning(f"Could not load all models: {e}")
        return None

@st.cache_resource
def load_shap_explainer():
    """Load SHAP explainer"""
    try:
        with open('../models/shap_explainer.pkl', 'rb') as f:
            shap_data = pickle.load(f)
        return shap_data
    except Exception as e:
        st.warning(f"SHAP explainer not available: {e}")
        return None

@st.cache_data
def load_ensemble_info():
    """Load ensemble model info"""
    try:
        with open('../models/ensemble_info.pkl', 'rb') as f:
            ensemble_info = pickle.load(f)
        return ensemble_info
    except Exception as e:
        st.warning("Ensemble info not available")
        return None

@st.cache_data
def load_validation_data():
    """Load validation data for metrics display"""
    try:
        X_val = pd.read_csv('../data/X_val.csv')
        y_val = pd.read_csv('../data/y_val.csv').values.ravel()
        return X_val, y_val
    except Exception as e:
        st.error(f"Error loading validation data: {e}")
        return None, None

@st.cache_data
def load_comparison_metrics():
    """Load model comparison results"""
    try:
        comparison_df = pd.read_csv('../reports/models_comparison_table.csv')
        return comparison_df
    except Exception as e:
        st.warning("Comparison metrics not available")
        return None

# ============================================
# MAIN HEADER
# ============================================

st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR - MODEL INFO & NAVIGATION
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-cards.png", width=100)
    st.title("🎯 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home & Predict", "🤖 Ensemble Voting", "🧠 Explainability (SHAP)", "📊 Model Performance", "📈 Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This system uses **Random Forest** machine learning model to detect fraudulent credit card transactions in real-time.
    
    **Accuracy:** High precision fraud detection
    **Model:** Random Forest Classifier
    **Dataset:** 284,807 transactions
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Team")
    st.markdown("**GIKI - AI Project**")
    st.markdown("Advanced AI Course")

# ============================================
# LOAD MODELS AND DATA
# ============================================

model = load_model()
all_models = load_all_models()
shap_data = load_shap_explainer()
ensemble_info = load_ensemble_info()
X_val, y_val = load_validation_data()

if model is None:
    st.error("❌ Failed to load model. Please check if the model file exists.")
    st.stop()

# ============================================
# PAGE 1: HOME & PREDICT
# ============================================

if page == "🏠 Home & Predict":
    
    st.markdown('<h2 class="sub-header">🔍 Transaction Fraud Detection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🧪 Quick Test")
        st.markdown("Try sample transactions:")
        
        if st.button("🟢 Test Legitimate Transaction", use_container_width=True):
            # Sample legitimate transaction (mostly zeros for V features)
            st.session_state['test_data'] = {
                'Time_Scaled': [0.5],
                'Amount_Scaled': [0.2],
                **{f'V{i}': [0.1] for i in range(1, 29)}
            }
            st.session_state['test_label'] = "Legitimate Sample"
        
        if st.button("🔴 Test Fraud Transaction", use_container_width=True):
            # Sample fraud transaction (pattern from actual fraud)
            st.session_state['test_data'] = {
                'Time_Scaled': [-0.8],
                'Amount_Scaled': [1.5],
                'V1': [-2.31], 'V2': [1.95], 'V3': [-1.60], 'V4': [3.99],
                'V5': [-0.52], 'V6': [-1.42], 'V7': [-2.53], 'V8': [1.39],
                'V9': [-2.77], 'V10': [-2.77], 'V11': [3.20], 'V12': [-2.99],
                'V13': [-0.59], 'V14': [-4.29], 'V15': [0.39], 'V16': [-1.42],
                'V17': [-2.83], 'V18': [-0.01], 'V19': [0.42], 'V20': [0.50],
                'V21': [0.22], 'V22': [0.66], 'V23': [-0.17], 'V24': [0.36],
                'V25': [0.00], 'V26': [-0.08], 'V27': [0.01], 'V28': [0.01]
            }
            st.session_state['test_label'] = "Fraud Sample"
    
    with col1:
        st.markdown("### 📝 Enter Transaction Details")
        
        # Check if test data exists in session state
        if 'test_data' in st.session_state:
            default_data = st.session_state['test_data']
            st.info(f"✅ Loaded: **{st.session_state['test_label']}**")
        else:
            default_data = None
        
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                time_scaled = st.number_input(
                    "Time (Scaled)", 
                    value=float(default_data['Time_Scaled'][0]) if default_data else 0.0,
                    format="%.4f",
                    help="Normalized time of transaction"
                )
                amount_scaled = st.number_input(
                    "Amount (Scaled)", 
                    value=float(default_data['Amount_Scaled'][0]) if default_data else 0.0,
                    format="%.4f",
                    help="Normalized transaction amount"
                )
            
            with col_b:
                st.markdown("**PCA Features (V1-V28)**")
                st.caption("Principal components from PCA transformation")
            
            # Expandable section for V features
            with st.expander("🔧 Configure V Features (V1-V28)", expanded=False):
                v_features = {}
                cols = st.columns(4)
                for i in range(1, 29):
                    with cols[(i-1) % 4]:
                        v_features[f'V{i}'] = st.number_input(
                            f'V{i}', 
                            value=float(default_data[f'V{i}'][0]) if default_data else 0.0,
                            format="%.4f",
                            key=f'v{i}'
                        )
            
            submitted = st.form_submit_button("🔮 Predict Transaction", use_container_width=True)
            
            if submitted:
                # Prepare input data - match exact training order
                try:
                    X_train_sample = pd.read_csv('../data/X_train_smote.csv', nrows=1)
                    feature_order = X_train_sample.columns.tolist()
                except:
                    # Fallback order if file not found
                    feature_order = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                                     'Amount_Scaled', 'Time_Scaled']
                
                # Create input with all features
                input_dict = {
                    'Time_Scaled': time_scaled,
                    'Amount_Scaled': amount_scaled,
                    **v_features
                }
                
                # Create DataFrame with correct order
                input_data = pd.DataFrame([input_dict])[feature_order]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown('<h3 class="sub-header">🎯 Prediction Result</h3>', unsafe_allow_html=True)
                
                # Display result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <h2 style="color: #e74c3c; margin: 0;">⚠️ FRAUD DETECTED</h2>
                        <p style="font-size: 1.2rem; margin: 10px 0;">This transaction is likely <strong>FRAUDULENT</strong></p>
                        <p style="font-size: 1rem;">Fraud Confidence: <strong>{probability[1]*100:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h2 style="color: #2ecc71; margin: 0;">✅ LEGITIMATE</h2>
                        <p style="font-size: 1.2rem; margin: 10px 0;">This transaction appears <strong>LEGITIMATE</strong></p>
                        <p style="font-size: 1rem;">Legitimacy Confidence: <strong>{probability[0]*100:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability gauge
                st.markdown("### 📊 Confidence Score")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability", 'font': {'size': 24}},
                    delta = {'reference': 50, 'increasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 70], 'color': '#fff3cd'},
                            {'range': [70, 100], 'color': '#f8d7da'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Probability breakdown
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
                with col_p2:
                    st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")

# ============================================
# PAGE 2: ENSEMBLE VOTING
# ============================================

elif page == "🤖 Ensemble Voting":
    
    st.markdown('<h2 class="sub-header">🤖 Ensemble Voting System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is Ensemble Voting?
    
    Instead of relying on a single model, our system combines predictions from **4 different algorithms**:
    - 🔹 Logistic Regression (Linear baseline)
    - 🌲 Random Forest (Tree ensemble)
    - 🎯 Support Vector Machine (Kernel-based)
    - 🚀 XGBoost (Gradient boosting)
    
    Each model "votes" and we use **weighted soft voting** (probability-based) where better-performing models have more influence.
    """)
    
    if all_models is None or ensemble_info is None:
        st.error("Ensemble models not available. Please run notebook 07 first.")
    else:
        # Show ensemble performance
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{ensemble_info['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{ensemble_info['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{ensemble_info['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{ensemble_info['f1_score']:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{ensemble_info['roc_auc']:.4f}")
        
        st.markdown("---")
        
        # Interactive ensemble prediction
        st.markdown("### 🧪 Test Ensemble Voting")
        
        col_test1, col_test2 = st.columns([2, 1])
        
        with col_test2:
            st.markdown("#### Quick Test")
            if st.button("🟢 Test Legitimate", key="ensemble_legit", use_container_width=True):
                st.session_state['ensemble_test'] = {
                    'Time_Scaled': [0.5],
                    'Amount_Scaled': [0.2],
                    **{f'V{i}': [0.1] for i in range(1, 29)}
                }
            
            if st.button("🔴 Test Fraud", key="ensemble_fraud", use_container_width=True):
                st.session_state['ensemble_test'] = {
                    'Time_Scaled': [-0.8],
                    'Amount_Scaled': [1.5],
                    'V1': [-2.31], 'V2': [1.95], 'V3': [-1.60], 'V4': [3.99],
                    'V5': [-0.52], 'V6': [-1.42], 'V7': [-2.53], 'V8': [1.39],
                    'V9': [-2.77], 'V10': [-2.77], 'V11': [3.20], 'V12': [-2.99],
                    'V13': [-0.59], 'V14': [-4.29], 'V15': [0.39], 'V16': [-1.42],
                    'V17': [-2.83], 'V18': [-0.01], 'V19': [0.42], 'V20': [0.50],
                    'V21': [0.22], 'V22': [0.66], 'V23': [-0.17], 'V24': [0.36],
                    'V25': [0.00], 'V26': [-0.08], 'V27': [0.01], 'V28': [0.01]
                }
        
        with col_test1:
            if st.button("🔮 Run Ensemble Prediction", use_container_width=True, type="primary"):
                if 'ensemble_test' in st.session_state:
                    # Prepare input
                    try:
                        X_train_sample = pd.read_csv('../data/X_train_smote.csv', nrows=1)
                        feature_order = X_train_sample.columns.tolist()
                    except:
                        feature_order = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                         'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                         'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                                         'Amount_Scaled', 'Time_Scaled']
                    
                    input_dict = st.session_state['ensemble_test']
                    input_data = pd.DataFrame([input_dict])[feature_order]
                    
                    # Get predictions from all models
                    predictions = {}
                    probabilities = {}
                    
                    for name, model_obj in [('Logistic Regression', all_models['lr']),
                                           ('Random Forest', all_models['rf']),
                                           ('SVM', all_models['svm']),
                                           ('XGBoost', all_models['xgb'])]:
                        predictions[name] = model_obj.predict(input_data)[0]
                        probabilities[name] = model_obj.predict_proba(input_data)[0]
                    
                    # Calculate weighted ensemble
                    weights = ensemble_info['weights']
                    weighted_proba = (
                        weights[0] * probabilities['Logistic Regression'] +
                        weights[1] * probabilities['Random Forest'] +
                        weights[2] * probabilities['SVM'] +
                        weights[3] * probabilities['XGBoost']
                    ) / sum(weights)
                    
                    ensemble_pred = 1 if weighted_proba[1] >= 0.5 else 0
                    
                    st.markdown("---")
                    st.markdown("### 🗳️ Voting Results")
                    
                    # Show individual votes
                    vote_cols = st.columns(4)
                    model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost']
                    
                    for idx, (col, name) in enumerate(zip(vote_cols, model_names)):
                        with col:
                            vote = "FRAUD" if predictions[name] == 1 else "LEGIT"
                            color = "🔴" if predictions[name] == 1 else "🟢"
                            confidence = probabilities[name][1] * 100
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; background: rgba(30,30,50,0.6); border-radius: 10px; border: 2px solid {'#ef4444' if predictions[name] == 1 else '#22c55e'};">
                                <h4>{color} {vote}</h4>
                                <p style="font-size: 0.9rem;">Weight: {weights[idx]}</p>
                                <p style="font-size: 0.8rem;">Fraud: {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Final ensemble decision
                    if ensemble_pred == 1:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <h2 style="color: #e74c3c; margin: 0;">⚠️ ENSEMBLE: FRAUD DETECTED</h2>
                            <p style="font-size: 1.2rem; margin: 10px 0;">Weighted Confidence: <strong>{weighted_proba[1]*100:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h2 style="color: #2ecc71; margin: 0;">✅ ENSEMBLE: LEGITIMATE</h2>
                            <p style="font-size: 1.2rem; margin: 10px 0;">Weighted Confidence: <strong>{weighted_proba[0]*100:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Voting breakdown chart
                    st.markdown("### 📊 Probability Breakdown")
                    
                    fig = go.Figure()
                    
                    for name, weight in zip(model_names, weights):
                        fig.add_trace(go.Bar(
                            name=name,
                            x=['Legitimate', 'Fraud'],
                            y=probabilities[name] * 100,
                            text=[f'{p:.1f}%' for p in probabilities[name] * 100],
                            textposition='auto',
                        ))
                    
                    fig.update_layout(
                        title='Individual Model Probabilities',
                        xaxis_title='Prediction',
                        yaxis_title='Confidence (%)',
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("Please click a test button first!")
        
        # Comparison table
        st.markdown("---")
        st.markdown("### 📊 Ensemble vs Individual Models")
        
        try:
            comparison_df = pd.read_csv('../reports/ensemble_comparison.csv')
            st.dataframe(comparison_df, use_container_width=True)
        except:
            st.info("Comparison data not available")

# ============================================
# PAGE 3: EXPLAINABILITY (SHAP)
# ============================================

elif page == "🧠 Explainability (SHAP)":
    
    st.markdown('<h2 class="sub-header">🧠 Model Explainability with SHAP</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is SHAP?
    
    **SHAP (SHapley Additive exPlanations)** explains individual predictions by showing:
    - Which features pushed the prediction toward fraud/legitimate
    - By how much each feature contributed
    - Why the model made this specific decision
    
    This builds **trust** and helps understand the model's reasoning.
    """)
    
    if shap_data is None:
        st.error("SHAP explainer not available. Please run notebook 08 first.")
    else:
        tab1, tab2, tab3 = st.tabs(["🔍 Explain Prediction", "📊 Global Importance", "📈 Feature Analysis"])
        
        # Tab 1: Explain specific prediction
        with tab1:
            st.markdown("### 🔍 Explain a Specific Transaction")
            
            col_shap1, col_shap2 = st.columns([2, 1])
            
            with col_shap2:
                st.markdown("#### Sample Transactions")
                
                # Get sample indices
                X_sample = shap_data['X_sample']
                shap_values = shap_data['shap_values']
                explainer = shap_data['explainer']
                
                if st.button("🔴 Explain Fraud", use_container_width=True):
                    # Find a fraud example from X_sample
                    sample_indices = X_sample.index.tolist()
                    y_sample = y_val[sample_indices]
                    fraud_indices = np.where(y_sample == 1)[0]
                    
                    if len(fraud_indices) > 0:
                        st.session_state['explain_idx'] = fraud_indices[0]
                        st.session_state['explain_type'] = 'fraud'
                    else:
                        st.warning("No fraud samples in SHAP data")
                
                if st.button("🟢 Explain Legitimate", use_container_width=True):
                    sample_indices = X_sample.index.tolist()
                    y_sample = y_val[sample_indices]
                    legit_indices = np.where(y_sample == 0)[0]
                    
                    if len(legit_indices) > 0:
                        st.session_state['explain_idx'] = legit_indices[0]
                        st.session_state['explain_type'] = 'legitimate'
            
            with col_shap1:
                if 'explain_idx' in st.session_state:
                    idx = st.session_state['explain_idx']
                    exp_type = st.session_state['explain_type']
                    
                    st.info(f"Explaining **{exp_type.upper()}** transaction (index {idx})")
                    
                    # Get SHAP values for this prediction
                    shap_vals = shap_values[idx]
                    features = X_sample.iloc[idx]
                    
                    # Create waterfall data
                    import shap
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1]
                    
                    # Get top contributing features - safe version
                    try:
                     shap_vals_flat = np.ravel(shap_vals)
                     features_flat = np.ravel(features.values)
    
                     # Ensure all same length
                     n_features = len(X_sample.columns)
    
                     feature_contributions = pd.DataFrame({
                      'Feature': X_sample.columns.tolist(),
                      'Value': features_flat[:n_features] if len(features_flat) >= n_features else list(features_flat) + [0]*(n_features-len(features_flat)),
                      'SHAP': shap_vals_flat[:n_features] if len(shap_vals_flat) >= n_features else list(shap_vals_flat) + [0]*(n_features-len(shap_vals_flat))
                     })
                    except Exception as e:
                     st.error(f"Error creating feature contributions: {e}")
                     feature_contributions = pd.DataFrame()
                    feature_contributions['Abs_SHAP'] = np.abs(feature_contributions['SHAP'])
                    feature_contributions = feature_contributions.sort_values('Abs_SHAP', ascending=False)
                    
                    st.markdown("#### 📊 Top Feature Contributions")
                    
                    top_10 = feature_contributions.head(10)
                    
                    # Create bar chart
                    fig = go.Figure()
                    
                    colors = ['#ef4444' if x > 0 else '#22c55e' for x in top_10['SHAP']]
                    
                    fig.add_trace(go.Bar(
                        y=top_10['Feature'],
                        x=top_10['SHAP'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f'{x:.4f}' for x in top_10['SHAP']],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title='SHAP Values (Red = Push toward Fraud, Green = Push toward Legitimate)',
                        xaxis_title='SHAP Value',
                        yaxis_title='Features',
                        height=500,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation text
                    st.markdown("#### 💡 Interpretation")
                    
                    top_fraud_features = feature_contributions[feature_contributions['SHAP'] > 0].head(3)
                    top_legit_features = feature_contributions[feature_contributions['SHAP'] < 0].head(3)
                    
                    if len(top_fraud_features) > 0:
                        st.markdown("**Features pushing toward FRAUD:**")
                        for _, row in top_fraud_features.iterrows():
                            st.markdown(f"- **{row['Feature']}** (value: {row['Value']:.4f}) increased fraud probability by {row['SHAP']:.4f}")
                    
                    if len(top_legit_features) > 0:
                        st.markdown("**Features pushing toward LEGITIMATE:**")
                        for _, row in top_legit_features.iterrows():
                            st.markdown(f"- **{row['Feature']}** (value: {row['Value']:.4f}) decreased fraud probability by {abs(row['SHAP']):.4f}")
                else:
                    st.info("👈 Click a button to explain a transaction")
        
        # Tab 2: Global importance
        with tab2:
            st.markdown("### 📊 Global Feature Importance")
            
            st.markdown("""
            This shows which features are **most important overall** for fraud detection across all transactions.
            """)
            
            try:
                st.image('../reports/shap_global_importance.png', use_column_width=True)
                st.caption("SHAP Summary Plot - Shows feature importance and impact direction")
            except:
                st.warning("Global importance plot not available")
            
            try:
                st.image('../reports/shap_bar_chart.png', use_column_width=True)
                st.caption("Mean absolute SHAP values for each feature")
            except:
                pass
            
            # Show table
            try:
                shap_importance = pd.read_csv('../reports/shap_feature_importance.csv')
                st.markdown("#### Top 20 Features by Importance")
                st.dataframe(shap_importance.head(20), use_container_width=True)
            except:
                st.info("SHAP importance table not available")
        
        # Tab 3: Feature analysis
        with tab3:
            st.markdown("### 📈 Feature Dependence Analysis")
            
            st.markdown("""
            These plots show how individual feature values affect predictions.
            """)
            
            # Try to load dependence plots
            try:
                shap_importance = pd.read_csv('../reports/shap_feature_importance.csv')
                top_features = shap_importance.head(2)['Feature'].tolist()
                
                for feature in top_features:
                    try:
                        st.markdown(f"#### {feature}")
                        st.image(f'../reports/shap_dependence_{feature}.png', use_column_width=True)
                        st.markdown("---")
                    except:
                        st.info(f"Dependence plot for {feature} not available")
            except:
                st.info("Feature dependence plots not available")

# ============================================
# PAGE 4: MODEL PERFORMANCE
# ============================================

elif page == "📊 Model Performance":
    
    st.markdown('<h2 class="sub-header">📊 Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    if X_val is not None and y_val is not None:
        # Make predictions on validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        # Display metrics
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}", delta=f"{(accuracy-0.5):.2%}")
        with col2:
            st.metric("Precision", f"{precision:.4f}", help="% of predicted frauds that are actual frauds")
        with col3:
            st.metric("Recall", f"{recall:.4f}", help="% of actual frauds detected")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}", help="Harmonic mean of precision and recall")
        with col5:
            st.metric("ROC-AUC", f"{roc_auc:.4f}", help="Area under ROC curve")
        
        st.markdown("---")
        
        # Confusion Matrix
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            st.markdown("### 📋 Confusion Matrix")
            cm = confusion_matrix(y_val, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Legitimate', 'Fraud'],
                y=['Legitimate', 'Fraud'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                showscale=True
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=400
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Detailed breakdown
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            **Detailed Metrics:**
            - True Negatives: **{tn:,}** (Correctly identified legitimate)
            - False Positives: **{fp:,}** (Legitimate flagged as fraud)
            - False Negatives: **{fn:,}** (Fraud missed)
            - True Positives: **{tp:,}** (Correctly detected fraud)
            """)
        
        with col_cm2:
            st.markdown("### 📈 ROC Curve")
            from sklearn.metrics import roc_curve
            
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Random Forest (AUC = {roc_auc:.4f})',
                line=dict(color='#e74c3c', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Model Comparison
        st.markdown("---")
        st.markdown("### 🏆 Model Comparison")
        
        comparison_df = load_comparison_metrics()
        if comparison_df is not None:
            st.dataframe(comparison_df, use_container_width=True)
            
            # Bar chart comparison
            fig_comp = go.Figure()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            for metric in metrics:
                fig_comp.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(4),
                    textposition='auto',
                ))
            
            fig_comp.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
    
    else:
        st.error("Validation data not available")

# ============================================
# PAGE 5: ANALYTICS
# ============================================

elif page == "📈 Analytics":
    
    st.markdown('<h2 class="sub-header">📈 Feature Importance & Insights</h2>', unsafe_allow_html=True)
    
    # Feature Importance
    try:
        feature_importance = pd.DataFrame({
            'Feature': X_val.columns if X_val is not None else [],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.markdown("### 🔍 Top 20 Most Important Features")
        
        top_20 = feature_importance.head(20)
        
        fig_feat = go.Figure(go.Bar(
            x=top_20['Importance'],
            y=top_20['Feature'],
            orientation='h',
            marker=dict(
                color=top_20['Importance'],
                colorscale='Reds',
                showscale=True
            ),
            text=top_20['Importance'].round(4),
            textposition='auto'
        ))
        
        fig_feat.update_layout(
            title='Feature Importance (Random Forest)',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig_feat, use_container_width=True)
        
        # Feature importance table
        st.markdown("### 📋 Complete Feature Importance Table")
        st.dataframe(feature_importance, use_container_width=True, height=400)
        
        # Download button
        csv = feature_importance.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance CSV",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error displaying feature importance: {e}")
    
    # Project insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    col_i1, col_i2, col_i3 = st.columns(3)
    
    with col_i1:
        st.info("""
        **Class Imbalance Handling**
        
        Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset, creating synthetic fraud samples for better model training.
        """)
    
    with col_i2:
        st.success("""
        **Model Selection**
        
        Random Forest outperformed other models due to its ensemble nature and ability to handle complex non-linear relationships in fraud patterns.
        """)
    
    with col_i3:
        st.warning("""
        **Feature Engineering**
        
        PCA-transformed features (V1-V28) preserve privacy while maintaining fraud detection capability. Amount and Time features were normalized for better model performance.
        """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p><strong>Credit Card Fraud Detection System</strong></p>
    <p>Developed as part of Advanced Artificial Intelligence Course | GIKI</p>
    <p>Powered by Random Forest Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)