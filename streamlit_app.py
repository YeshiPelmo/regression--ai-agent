import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os


st.set_page_config(layout='wide')
#🌟Title
st.title("💼 Insurance Cost Predictor: A Regression-Based Approach")
st.markdown("""
Welcome to the **Insurance Cost Predictor**, an advanced machine learning tool designed to estimate insurance charges based on personal health and demographic factors.
""")
#uploading data source

@st.cache_data
def load_data():
 path=r"C:\Users\tlhun\OneDrive\Desktop\Baruch\DataMining\Project 1\regression--ai-agent\data\raw\insurance.csv"
 df = pd.read_csv(path)
 return df

df = load_data()

#Display data 
st.subheader ("🔎 Preview of 100 sample Data")
st.write(df.head(100))

#Show basic info
st.subheader("📊 Data Summary")
st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
st.markdown(f"**Columns:** {', '.join(df.columns)}")

# Descriptive stats
st.subheader("📈 Statistical Overview")
st.dataframe(df.describe().T)

# --- Encode Categorical Columns ---
df_numeric = df.copy()
le = LabelEncoder()
categorical_cols = ['sex', 'smoker', 'region']

for col in categorical_cols:
    if col in df_numeric.columns:
        df_numeric[f'{col}_encoded'] = le.fit_transform(df_numeric[col])
        df_numeric = df_numeric.drop(col, axis=1)

# --- Generate Correlation Heatmap ---
correlation_matrix = df_numeric.corr()

st.subheader("📊 Feature Correlation Matrix")

fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    annot_kws={"size": 10},
    ax=ax
)

plt.title('Feature Correlation Matrix of Insurance Charges')
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot(fig)

required_files = [
    "best_model.joblib",
    "feature_names.joblib",
    "model_metadata.joblib",
    "scaler.joblib"
]   
# Load model and metadata
@st.cache_resource
def load_model_artifacts():
    """Load the trained model artifacts."""
    try:
        model_path = "models/best_model.joblib"
        feature_names_path = "models/feature_names.joblib"
        metadata_path = "models/model_metadata.joblib"
        scaler_path = "models/scaler.joblib"

        if all(os.path.exists(p) for p in [model_path, feature_names_path, metadata_path, scaler_path]):
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_names_path)
            metadata = joblib.load(metadata_path)
            scaler = joblib.load(scaler_path)
            return model, feature_names, metadata, scaler
        else:
            st.error("❌ Model files not found! Please ensure the models/ directory contains your saved model files.")
            return None, None, None, None

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, 


def create_features(age, sex, bmi, children, smoker, region):
    """
    Create feature vector matching your training data structure
    with proper feature engineering and one-hot encoding
    """
    # Create BMI categories (matching your bins)
    if bmi <= 18.5:
        bmi_category = 'underweight'
    elif bmi <= 25:
        bmi_category = 'normal'
    elif bmi <= 30:
        bmi_category = 'overweight'
    else:
        bmi_category = 'obese'
    
    # Create age groups (matching your bins)
    if age <= 18:
        age_group = 'child'
    elif age <= 35:
        age_group = 'young_adult'
    elif age <= 50:
        age_group = 'middle_aged'
    elif age <= 65:
        age_group = 'senior'
    else:
        age_group = 'elderly'
    
    # Create the feature vector matching your exact structure
    features = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
        'bmi_category_normal': 1 if bmi_category == 'normal' else 0,
        'bmi_category_overweight': 1 if bmi_category == 'overweight' else 0,
        'bmi_category_obese': 1 if bmi_category == 'obese' else 0,
        'age_group_young_adult': 1 if age_group == 'young_adult' else 0,
        'age_group_middle_aged': 1 if age_group == 'middle_aged' else 0,
        'age_group_senior': 1 if age_group == 'senior' else 0,
        'age_group_elderly': 1 if age_group == 'elderly' else 0
    }
    
    return pd.DataFrame([features]), bmi_category, age_group

def make_prediction(model, scaler, age, sex, bmi, children, smoker, region):
    """Make prediction using your trained Random Forest model with exact preprocessing"""
    if model is None or scaler is None:
        return 0, None, None
    
    try:
        # Create features with proper engineering
        features_df, bmi_category, age_group = create_features(age, sex, bmi, children, smoker, region)
        
        # Apply exact scaling using your saved scaler
        scaled_features = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        return max(0, prediction), bmi_category, age_group
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return 0, None, None
    
# Load model and artifacts
model, feature_names, metadata, scaler = load_model_artifacts()

# Display model information
if metadata:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy (R²)", f"{metadata['r2']:.1%}")
    with col2:
        st.metric("RMSE", f"${metadata['rmse']:,.0f}")
    with col3:
        st.metric("MAE", f"${metadata['mae']:,.0f}")
    with col4:
        st.metric("Algorithm", "Random Forest")

# Sidebar for input parameters
st.sidebar.markdown('<h2 class="sub-header">📊 Input Parameters</h2>', unsafe_allow_html=True)

# Input widgets
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30, 
                       help="Age of the person in years")

sex = st.sidebar.selectbox("Sex", options=["male", "female"], 
                          help="Gender of the person")

bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                             help="Body Mass Index (weight/height²)")

children = st.sidebar.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5],
                               help="Number of children covered by insurance")

smoker = st.sidebar.selectbox("Smoker", options=["no", "yes"],
                             help="Whether the person is a smoker")

region = st.sidebar.selectbox("Region", 
                             options=["northeast", "northwest", "southeast", "southwest"],
                             help="Region where the person lives")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">🔮 Prediction Results</h2>', unsafe_allow_html=True)
    
    if st.button("💰 Predict Insurance Charges", type="primary", use_container_width=True):
        # Make prediction
        predicted_charges, bmi_category, age_group = make_prediction(model, scaler, age, sex, bmi, children, smoker, region)
        
        if predicted_charges > 0:
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>💰 Predicted Annual Insurance Charges</h3>
                <h2 style="color: #1f77b4; font-size: 2.5rem;">${predicted_charges:,.2f}</h2>
                <p style="color: #666; margin-top: 1rem;">
                    🎯 Prediction using your exact Random Forest model & preprocessing pipeline
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model insights based on your actual feature importance
            st.markdown("### 🎯 Feature Importance from Your Model")
            st.info(f"""
            **Top 5 Most Important Factors (from your Random Forest):**
            1. **Smoking Status** - 60.86% importance
            2. **BMI** - 15.26% importance  
            3. **Age** - 13.05% importance
            4. **BMI Category (Obese)** - 5.90% importance
            5. **Children** - 1.90% importance
            
            Your input profile: {age_group.replace('_', ' ').title()}, BMI Category: {bmi_category.title()}
            """)
with col2:
    st.markdown('<h2 class="sub-header">📋 Input Summary</h2>', unsafe_allow_html=True)
    
    # Get BMI category for display
    if bmi <= 18.5:
        bmi_category_display = "Underweight"
        bmi_color = "#3498db"
    elif bmi <= 25:
        bmi_category_display = "Normal"
        bmi_color = "#2ecc71"
    elif bmi <= 30:
        bmi_category_display = "Overweight"
        bmi_color = "#f39c12"
    else:
        bmi_category_display = "Obese"
        bmi_color = "#e74c3c"            

# Display current inputs
    st.markdown(f"""
    <div class="metric-card">
        <strong>👤 Personal Info:</strong><br>
        • Age: {age} years<br>
        • Sex: {sex.title()}<br>
        • Children: {children}<br><br>
        
        <strong>🏥 Health Info:</strong><br>
        • BMI: {bmi}<br>
        • Category: <span style="color: {bmi_color}; font-weight: bold;">{bmi_category_display}</span><br>
        • Smoker: {smoker.title()}<br><br>
        
        <strong>📍 Location:</strong><br>
        • Region: {region.title()}
    </div>
    """, unsafe_allow_html=True)
 
# Risk level indicator
    risk_score = 0
    if smoker == "yes":
        risk_score += 3
    if bmi > 30:
        risk_score += 2
    elif bmi < 18.5:
        risk_score += 1
    if age > 50:
        risk_score += 2
    if age > 65:
        risk_score += 1
        
    if risk_score >= 5:
        risk_level = "High"
        risk_color = "#e74c3c"
    elif risk_score >= 3:
        risk_level = "Moderate"
        risk_color = "#f39c12"
    else:
        risk_level = "Low"
        risk_color = "#2ecc71"
        
    st.markdown(f"""
    <div class="metric-card">
        <strong>⚡ Risk Assessment:</strong><br>
        <span style="color: {risk_color}; font-size: 1.3rem; font-weight: bold;">{risk_level} Risk Profile</span>
    </div>
    """, unsafe_allow_html=True)
    
 
# Model Performance Section
st.markdown("---")
st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)

if metadata:
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
        **🎯 Model Accuracy**
        - **R² Score**: 86.6%
        - **RMSE**: $4,555
        - **MAE**: $2,565
        - **Algorithm**: Random Forest
        """)
    
    with col4:
        st.markdown("""
        **📊 Top Features (Importance)**
        - **Smoking**: 60.9%
        - **BMI**: 15.3%
        - **Age**: 13.1%
        - **BMI Category**: 5.9%
        - **Children**: 1.9%
        """)
    
    with col5:
        st.markdown("""
        **🔧 Model Details**
        - **Dataset**: Insurance Charges (Kaggle)
        - **Features**: 15 engineered features
        - **Training**: 1,070 samples
        - **Testing**: 268 samples
        """)

# Instructions for usage
st.markdown("---")
st.markdown("### 💡 How to Use This Predictor")

with st.expander("Click here for detailed instructions"):
    st.markdown("""
    **🔍 Understanding Your Prediction:**
    
    1. **Input Your Information**: Use the sidebar to enter your personal details
    2. **Click Predict**: The model will analyze your profile using 15 different factors
    3. **Review Results**: See your predicted annual insurance charges and risk analysis
    
    **📈 Key Factors That Affect Your Premium:**
    
    - **Smoking** (60.9% importance): The single biggest factor - smokers pay significantly more
    - **BMI** (15.3% importance): Both obesity and being underweight can increase costs
    - **Age** (13.1% importance): Older individuals typically have higher premiums
    - **Children**: More dependents can increase family coverage costs
    - **Region**: Different areas have varying healthcare costs
    
    **⚠️ Important Notes:**
    
    - This is a predictive model for educational purposes
    - Actual insurance quotes may vary based on additional factors
    - The model was trained on historical data and may not reflect current market rates
    - Always consult with insurance providers for official quotes
    
    **🎯 Model Performance:**
    
    - **86.6% accuracy** on test data
    - **Trained on 1,338 insurance records**
    - **Uses advanced feature engineering** including BMI categories and age groups
    - **Random Forest algorithm** selected after comparing 4 different models
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Insurance Charges Predictor</strong> | Built with Streamlit & Random Forest ML</p>
    <p>Model Performance: 86.6% R² Score | RMSE: $4,555 | Based on Kaggle Insurance Dataset</p>
</div>
""", unsafe_allow_html=True)                