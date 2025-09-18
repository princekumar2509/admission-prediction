import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# --- Background Image Functions ---
import streamlit as st
import base64


st.set_page_config(
        page_title="Admission Predictor Pro", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🎓Admission Predictor"
    )

    
with open("cllg2.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Create CSS with the background image and styles
page_bg_img = f"""
<style>
.stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    filter: blur(10px);
    z-index: -1;
}}
.stApp {{
    background: none;
}}
.main .block-container {{
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 2.5rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.8);
}}
.stSelectbox > div > div {{
    background-color: white;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}}
.stTextInput > div > div > input {{
    background-color: white;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}}
.stFileUploader > div {{
    background-color: white;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}}
.stButton > button {{
    background-color: #ff6b6b;
    color: white;
    border-radius: 8px;
    border: none;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    font-weight: bold;
}}
.stButton > button:hover {{
    background-color: #ff5252;
    transform: translateY(-2px);
}}
.stRadio > div {{
    background-color: white;
    border-radius: 8px;
    padding: 1rem;
    border: 2px solid #e0e0e0;
}}
.stMultiSelect > div > div {{
    background-color: white;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}}
.stNumberInput > div > div > input {{
    background-color: white;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
}}
.stMetric {{
    background-color: white;
    border-radius: 10px;
    padding: 1rem;
    border: 2px solid #e0e0e0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}}
.stDataFrame {{
    background-color: white;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
}}
.stAlert {{
    background-color: white;
    border-radius: 10px;
    border: 2px solid #e0e0e0;
}}
.stSuccess {{
    background-color: #d4edda;
    border-radius: 10px;
    border: 2px solid #c3e6cb;
}}
.stWarning {{
    background-color: #fff3cd;
    border-radius: 10px;
    border: 2px solid #ffeaa7;
}}
.stError {{
    background-color: #f8d7da;
    border-radius: 10px;
    border: 2px solid #f5c6cb;
}}
.stInfo {{
    background-color: #d1ecf1;
    border-radius: 10px;
    border: 2px solid #bee5eb;
}}
h1, h2, h3, h4, h5, h6 {{
    color: white;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
}}
p, div {{
    color: #fc8c03;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df

@st.cache_data
def load_csv_from_path(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_csv_from_url(url):
    df = pd.read_csv(url)
    return df

def check_dataset(df):
    if df.empty:
        return "The dataset is empty."
    elif df.isnull().values.any():
        missing = df.isnull().sum()
        return f"The dataset has missing values:\n{missing[missing > 0]}"
    else:
        return "The dataset looks good!"

# --- Main Application ---
def main():
    menu = ["Home", "Predict", "Analytics", "About"]
    with st.container():
        cols = st.columns([8, 1, 1])
        with cols[0]:
            st.markdown("""
            <div style="font-size:48px;text-align:center; font-weight:bold; color:white; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                🎓 Admission Predictor
            </div>
            """, unsafe_allow_html=True)
        choice = cols[2].selectbox("Menu", menu, key="main_menu")

    if choice == "Home":
        home()
    elif choice == "Predict":
        predict_page()
    elif choice == "Analytics":
        analytics_page()
    elif choice == "About":
        about_page()


# --- Home Page ---
def home():
    st.markdown("""
    <h3 style="color: #4CAF50; font-size: 36px; font-weight: bold; margin-bottom: 10px;">
         Welcome to Admission Predictor...
    </h3>
    <p style="color: #03fce8; font-size: 18px;">
        Your comprehensive tool for predicting university admission chances
    </p>
    """, unsafe_allow_html=True)
    # Key features overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📊 Data Analysis**
        - Upload and analyze admission datasets
        - Visualize patterns and trends
        - Identify key factors affecting admission
        """)
    
    with col2:
        st.markdown("""
        **🤖 AI Prediction**
        - Advanced machine learning models
        - Individual student predictions
        - Real-time probability calculations
        """)
    
    with col3:
        st.markdown("""
        **📈 Analytics Dashboard**
        - Interactive charts and graphs
        - Performance metrics
        - Comparative analysis
        """)
    
    st.markdown("---")
    
    # Load default dataset
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px;  margin-bottom: 10px;">
        Load Dataset
    </h1>
    
    """, unsafe_allow_html=True)
    
    load_method = st.radio("Choose how to load the dataset", 
                          ["Use Sample Dataset", "Upload file", "From local path", "From URL"])

    df = None

    if load_method == "Use Sample Dataset":
        try:
            df = load_csv_from_path("adm_data.csv")
            st.success("✅ Sample dataset loaded successfully!")
        except:
            st.error("Sample dataset not found. Please upload your own dataset.")

    elif load_method == "Upload file":
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

    elif load_method == "From local path":
        path = st.text_input("Enter file path (e.g., data/myfile.csv)")
        if path:
            try:
                df = load_csv_from_path(path)
                st.success("✅ File loaded successfully!")
            except FileNotFoundError:
                st.error("File not found. Check the path.")

    elif load_method == "From URL":
        url = st.text_input("Enter CSV URL")
        if url:
            try:
                df = load_csv_from_url(url)
                st.success("✅ Dataset loaded from URL!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    if df is not None:
         st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px;  margin-bottom: 10px;">
        Dataset Overview
    </h1>
    
    """, unsafe_allow_html=True)
        
        # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
            st.metric("Total Records", len(df))
    with col2:
            st.metric("Features", len(df.columns))
    with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
            st.metric("Data Types", len(df.dtypes.unique()))
        
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px;  margin-bottom: 10px;">
        Sample Data
    </h1>
    
    """, unsafe_allow_html=True)
        
        
    st.dataframe(df.head(10), use_container_width=True)
        
        # Data quality check
    result = check_dataset(df)
    if "missing values" in result.lower():
            st.warning(f"⚠️ {result}")
    else:
            st.success(f"✅ {result}")
        
        # Store in session state for other pages
    st.session_state['dataset'] = df
    st.session_state['dataset_loaded'] = True
        
    if result == "The dataset looks good!":
            st.success("🎉 Dataset is ready for analysis! Navigate to 'Predict' or 'Analytics' to continue.")

# --- Prediction Page ---
def predict_page():
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        Admission Prediction
    </h1>
    """, unsafe_allow_html=True)

    if 'dataset_loaded' not in st.session_state or not st.session_state['dataset_loaded']:
        st.warning("⚠️ Please load a dataset from the Home page first.")
        return  # Correct indentation here

    df = st.session_state['dataset']

    # Model selection
    st.subheader("🤖 Select Model")
    model_type = st.selectbox("Choose ML Model", 
                              ["Logistic Regression", "Random Forest", "Both Models"])

    # Feature and target selection
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        Configure Model
    </h1>
    """, unsafe_allow_html=True)

    all_columns = df.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Select target column", all_columns)
    with col2:
        feature_columns = st.multiselect("Select feature columns", 
                                         [col for col in all_columns if col != target_column],
                                         default=[col for col in all_columns if col != target_column][:5])

    if st.button("🚀 Train Model", type="primary"):
        if not feature_columns:
            st.error("Please select at least one feature.")
            return
        train_advanced_model(df, feature_columns, target_column, model_type)

    # Individual prediction form
    st.markdown("---")
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        Individual Student Prediction
    </h1>
    """, unsafe_allow_html=True)

    if 'model_trained' in st.session_state and st.session_state['model_trained']:
        individual_prediction_form(df, st.session_state['feature_columns'], 
                                   st.session_state['target_column'], 
                                   st.session_state['model'])
    else:
        st.info("Please train a model first to make individual predictions.")


def train_advanced_model(df, features, target, model_type):
    st.write("🔄 Training model...")
    
    try:
        X = df[features]
        y = df[target]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        results = {}
        
        if model_type in ["Logistic Regression", "Both Models"]:
            # Logistic Regression
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_acc = accuracy_score(y_test, lr_pred)
            
            models['Logistic Regression'] = lr_model
            results['Logistic Regression'] = {
                'accuracy': lr_acc,
                'predictions': lr_pred,
                'scaler': scaler
            }
        
        if model_type in ["Random Forest", "Both Models"]:
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_pred)
            
            models['Random Forest'] = rf_model
            results['Random Forest'] = {
                'accuracy': rf_acc,
                'predictions': rf_pred,
                'scaler': None
            }
        
        # Display results
        st.success("✅ Model(s) trained successfully!")
        
        # Performance metrics
        st.subheader("📈 Model Performance")
        for model_name, result in results.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{model_name} Accuracy", f"{result['accuracy']:.3f}")
            with col2:
                st.metric("Test Samples", len(y_test))
            with col3:
                st.metric("Features Used", len(features))
        
        # Store in session state
        st.session_state['model'] = models
        st.session_state['feature_columns'] = features
        st.session_state['target_column'] = target
        st.session_state['model_trained'] = True
        st.session_state['results'] = results
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in models:
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': models['Random Forest'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title="Feature Importance (Random Forest)")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error training model: {e}")

def individual_prediction_form(df, features, target, models):
    st.write("Enter student details to predict admission chances:")
    
    # Create input form
    input_data = {}
    cols = st.columns(2)
    
    for i, feature in enumerate(features):
        with cols[i % 2]:
            if df[feature].dtype in ['int64', 'float64']:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=mean_val,
                    step=0.1
                )
            else:
                unique_vals = df[feature].unique()
                input_data[feature] = st.selectbox(f"{feature}", unique_vals)
    
    if st.button("🔮 Predict Admission Chance", type="primary"):
        # Make predictions
        input_df = pd.DataFrame([input_data])
        
        for model_name, model in models.items():
            try:
                if model_name == 'Logistic Regression':
                    # Use scaler for logistic regression
                    scaler = st.session_state['results'][model_name]['scaler']
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                else:
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**{model_name} Prediction:** {prediction}")
                with col2:
                    if len(probability) > 1:
                        max_prob = max(probability)
                        st.info(f"**Confidence:** {max_prob:.2%}")
                    else:
                        st.info(f"**Probability:** {probability[0]:.2%}")
                
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")

# --- Analytics Page ---
def analytics_page():
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        Data Analytics Dashboard...
    </h1>
    """, unsafe_allow_html=True)
    
    
    if 'dataset_loaded' not in st.session_state or not st.session_state['dataset_loaded']:
        st.warning("⚠️ Please load a dataset from the Home page first.")
        return
    
    df = st.session_state['dataset']
    
    # Data overview
    st.subheader("📈 Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Features", len(df.columns))
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Statistical summary
    st.subheader("📋 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        Data Visualizations...
    </h1>
    """, unsafe_allow_html=True)
    
    
    # Select columns for visualization
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) > 0:
        viz_type = st.selectbox("Select visualization type", 
                               ["Distribution", "Correlation", "Scatter Plot", "Box Plot"])
        
        if viz_type == "Distribution":
            selected_col = st.selectbox("Select column", numeric_columns)
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation":
            if len(numeric_columns) > 1:
                corr_matrix = df[numeric_columns].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        elif viz_type == "Scatter Plot":
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_columns)
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_columns, index=1)
                
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for scatter plot.")
        
        elif viz_type == "Box Plot":
            selected_col = st.selectbox("Select column for box plot", numeric_columns)
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Missing values analysis
    if df.isnull().sum().sum() > 0:
        st.subheader("🔍 Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)

# --- About Page ---
def about_page():
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        ℹ️ About Admission Predictor...
    </h1>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    
    
    **Admission Predictor Pro** is a comprehensive machine learning application designed to help students, 
    educators, and institutions predict university admission chances with high accuracy.
    """)
    
    # Features section
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        ✨ Key Features
    </h1>
    """, unsafe_allow_html=True)
    
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Advanced Machine Learning**
        - Multiple ML algorithms (Logistic Regression, Random Forest)
        - Real-time model training and evaluation
        - Feature importance analysis
        - Cross-validation and performance metrics
        
        **📊 Interactive Analytics**
        - Dynamic data visualizations
        - Statistical analysis and insights
        - Correlation matrices and distributions
        - Missing value analysis
        """)
    
    with col2:
        st.markdown("""
        **🎯 Individual Predictions**
        - Personalized admission probability
        - Real-time prediction interface
        - Confidence scores and explanations
        - Multiple model comparison
        
        **📈 Data Management**
        - Multiple data loading options
        - Data quality assessment
        - Preprocessing and feature engineering
        - Export and sharing capabilities
        """)
    
    # Technology stack
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        Technology Stack
    </h1>
    """, unsafe_allow_html=True)
    
    
    
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.markdown("**Frontend**\n- Streamlit\n- Plotly\n- HTML/CSS")
    with tech_cols[1]:
        st.markdown("**Backend**\n- Python\n- Pandas\n- NumPy")
    with tech_cols[2]:
        st.markdown("**ML Libraries**\n- Scikit-learn\n- Matplotlib\n- Seaborn")
    with tech_cols[3]:
        st.markdown("**Features**\n- Real-time predictions\n- Interactive charts\n- Responsive design")
    
    # How to use
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        How to Use
    </h1>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    1. **🏠 Home Page**: Load your dataset using the sample data or upload your own CSV file
    2. **🎯 Predict Page**: Train machine learning models and make individual predictions
    3. **📊 Analytics Page**: Explore your data with interactive visualizations and statistical analysis
    4. **ℹ️ About Page**: Learn more about the application and its capabilities
    """)
    
    # Contact/Info
    st.markdown("""
    <h1 style="color: #4CAF50; font-size: 36px; margin-bottom: 10px;">
        📞 Contact & Support
    </h1>
    """, unsafe_allow_html=True)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📧 Email**\ncontact@admissionpredictor.com")
    with col2:
        st.markdown("**🌐 Website**\nwww.admissionpredictor.com")
    with col3:
        st.markdown("**📱 Version**\n2.0.0 Pro")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2024 Admission Predictor Pro. Built with ❤️ using Streamlit and Python.</p>
        <p>Empowering students to make informed decisions about their academic future.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
