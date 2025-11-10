import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('DataSet/DataSet.csv')
    
    # Handle missing values
    df['bmi'] = df['bmi'].replace('N/A', np.nan)
    df['bmi'] = pd.to_numeric(df['bmi'])
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])
    
    return df

# Train model
@st.cache_data
def train_model(df):
    X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
    y = df['stroke']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Configure page
st.set_page_config(
    page_title="Stroke Prediction AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.risk-high {
    background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.risk-low {
    background: linear-gradient(90deg, #51cf66 0%, #40c057 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧠 AI Stroke Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Machine Learning for Healthcare Risk Assessment</p>', unsafe_allow_html=True)

# Add separator
st.markdown("---")

# Load data and train model
df = load_data()
model, accuracy = train_model(df)

# Enhanced sidebar
st.sidebar.markdown('<h2 style="color: #1f77b4;">📈 Model Analytics</h2>', unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="metric-card"><h3>Model Accuracy</h3><h2>{accuracy:.1%}</h2></div>', unsafe_allow_html=True)
st.sidebar.markdown("")

# Dataset info in sidebar
st.sidebar.markdown('<h3 style="color: #ff7f0e;">📊 Dataset Info</h3>', unsafe_allow_html=True)
st.sidebar.metric("Total Records", len(df))
st.sidebar.metric("Stroke Cases", df['stroke'].sum())
st.sidebar.metric("Stroke Rate", f"{df['stroke'].mean():.1%}")



# Input section with better styling
st.markdown('<h2 class="sub-header">📝 Patient Information Input</h2>', unsafe_allow_html=True)
st.markdown("Please fill in the patient details below for stroke risk assessment:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Prediction button with better styling
st.markdown("")
predict_button = st.button("🔮 Predict Stroke Risk", type="primary", use_container_width=True)

if predict_button:
    # Encode inputs
    gender_encoded = 1 if gender == "Male" else 0
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0
    
    work_type_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 4, "Never_worked": 1}
    work_type_encoded = work_type_map[work_type]
    
    residence_encoded = 1 if residence_type == "Urban" else 0
    
    smoking_map = {"never smoked": 2, "formerly smoked": 1, "smokes": 3, "Unknown": 0}
    smoking_encoded = smoking_map[smoking_status]
    
    # Make prediction
    input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
                           ever_married_encoded, work_type_encoded, residence_encoded, 
                           avg_glucose_level, bmi, smoking_encoded]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
    
    # Create columns for result display
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    
    with result_col2:
        if prediction == 1:
            st.markdown(f'<div class="risk-high"><h2>⚠️ HIGH RISK</h2><h3>{probability:.1%} Stroke Probability</h3></div>', unsafe_allow_html=True)
            st.error("🚨 **Recommendation:** Immediate medical consultation advised")
        else:
            st.markdown(f'<div class="risk-low"><h2>✅ LOW RISK</h2><h3>{probability:.1%} Stroke Probability</h3></div>', unsafe_allow_html=True)
            st.success("😊 **Recommendation:** Continue regular health monitoring")
    
    # Risk gauge
    st.markdown("### Risk Level Indicator")
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stroke Risk %"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Dataset overview section
st.markdown("---")
st.markdown('<h2 class="sub-header">📊 Dataset Overview & Model Insights</h2>', unsafe_allow_html=True)

# Create metrics columns
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="📁 Total Records",
        value=f"{len(df):,}",
        help="Total number of patient records in dataset"
    )

with metric_col2:
    st.metric(
        label="⚠️ Stroke Cases",
        value=f"{df['stroke'].sum():,}",
        help="Number of positive stroke cases"
    )

with metric_col3:
    st.metric(
        label="📈 Stroke Rate",
        value=f"{df['stroke'].mean():.1%}",
        help="Percentage of stroke cases in dataset"
    )

with metric_col4:
    st.metric(
        label="🎯 Model Accuracy",
        value=f"{accuracy:.1%}",
        help="Logistic regression model accuracy"
    )

