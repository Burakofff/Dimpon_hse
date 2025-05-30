import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

# Configure Streamlit page settings
st.set_page_config(
    page_title="Mental Health Assessment Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define mental health condition mapping
MENTAL_HEALTH_CLASSES = {0: 'satisfactory', 1: 'anxiety', 2: 'burnout'}

# Custom CSS styling for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model with caching
@st.cache_resource  
def load_model():
    """
    Load the trained model from pickle file.
    Tries multiple possible paths to find the model file.
    Returns the loaded model or None if loading fails.
    """
    try:
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir.parent / "model" / "model.pkl",
            current_dir / "model" / "model.pkl",
            Path("model") / "model.pkl",
            Path.cwd() / "model" / "model.pkl"
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
                
        if model_path is None:
            st.error("Model file not found. Please ensure the model file exists in one of these locations:")
            for path in possible_paths:
                st.error(f"- {path}")
            return None
            
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_model()
if model is None:
    st.error("Failed to load the model. Please check the error message above.")
    st.stop()

def predict(df):
    """
    Make predictions using the loaded model.
    Args:
        df: DataFrame with features
    Returns:
        Array of predictions or None if prediction fails
    """
    try:
        predictions = model.predict(df)
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def validate_input_data(df):
    """
    Validate input data for required columns and data types.
    Args:
        df: DataFrame to validate
    Returns:
        Boolean indicating if validation passed
    """
    required_columns = [
        "age", "gender", "years_of_experience", "work_location",
        "hours_worked_per_week", "number_of_virtual_meetings",
        "work_life_balance_rating", "stress_level",
        "access_to_mental_health_resources", "productivity_change",
        "social_isolation_rating", "satisfaction_with_remote_work",
        "company_support_for_remote_work", "physical_activity",
        "sleep_quality"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return False
        
    return True

def conv_cat_to_numeric(df):
    """
    Convert categorical variables to numeric format.
    Args:
        df: DataFrame with categorical variables
    Returns:
        DataFrame with converted numeric values
    """
    df_res = df.copy()

    # Map stress levels to numeric values
    stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df_res['stress_level'] = df_res['stress_level'].map(stress_mapping)

    # Map gender to numeric values
    gender_mapping = {'Male': 1, 'Female': 0}
    df_res['gender'] = df_res['gender'].map(gender_mapping).astype(int)

    # Map mental health resources access to numeric values
    stress_mapping = {'Yes': 1, 'No': 0}
    df_res['access_to_mental_health_resources'] = df_res['access_to_mental_health_resources'].map(
        stress_mapping).astype(int)

    # Map productivity change to numeric values
    mapping = {'Decrease': -1, 'No Change': 0, 'Increase': 1}
    df_res['productivity_change'] = df_res['productivity_change'].map(mapping).astype(int)

    # Map satisfaction with remote work to numeric values
    mapping = {'Unsatisfied': -1, 'Neutral': 0, 'Satisfied': 1}
    df_res['satisfaction_with_remote_work'] = df_res['satisfaction_with_remote_work'].map(mapping).astype(int)

    # Map work location to numeric values
    work_location_mapping = {'Onsite': 0, 'Hybrid': 1, 'Remote': 2}
    df_res['work_location'] = df_res['work_location'].map(work_location_mapping).astype(int)

    # Map physical activity to numeric values
    df_res['physical_activity'] = df_res['physical_activity'].fillna(0)
    physical_activity_mapping = {0: 0, 'Weekly': 1, 'Daily': 2}
    df_res['physical_activity'] = df_res['physical_activity'].fillna(0).map(physical_activity_mapping)

    # Map sleep quality to numeric values
    physical_activity_mapping = {'Average': 0, 'Poor': 1, 'Good': 2}
    df_res['sleep_quality'] = df_res['sleep_quality'].fillna(0).map(physical_activity_mapping)

    return df_res

def main():
    """
    Main function to run the Streamlit application.
    Handles both file upload and manual input scenarios.
    """
    st.title("Mental Health Assessment Tool")
    
    # Sidebar with application information
    with st.sidebar:
        st.header("About")
        st.write("""
        This tool helps to assess the mental health status of remote employees based on various factors.
        """)
        
        st.header("How to Use")
        st.write("""
        1. Choose input method (file upload or manual entry)
        2. Fill in all required information
        3. Get assessment results and recommendations
        """)
        
        st.header("Data Privacy")
        st.write("""
        Your data is processed locally and is not stored or shared.
        """)
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload file", "Enter manually"])

    if input_method == "Upload file":
        # File upload handling
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

        if uploaded_file:
            try:
                with st.spinner("Processing your file..."):
                    # Read uploaded file
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())

                    if validate_input_data(df):
                        # Process input data
                        df_input = df.copy()
                        df_input = conv_cat_to_numeric(df_input)

                        # Create derived features
                        df_input["age_exp_interaction"] = df_input["age"] * df_input["years_of_experience"]
                        df_input["workload"] = df_input["hours_worked_per_week"] * df_input["number_of_virtual_meetings"]
                        df_input["sleep_stress_ratio"] = round(df_input["sleep_quality"] / (df_input["stress_level"] + 1), 2)
                        df_input["physical_satisfaction"] = df_input["physical_activity"] * df_input["satisfaction_with_remote_work"]

                        # Handle infinite values in sleep_stress_ratio
                        df_input["sleep_stress_ratio"] = np.where(np.isinf(df_input["sleep_stress_ratio"]), 0,
                                                              df_input["sleep_stress_ratio"])
                        df_input["sleep_stress_ratio"] = round(df_input["sleep_stress_ratio"], 2)

                        # Select model features and scale data
                        df_input = df_input[model.feature_names_in_]
                        features = df_input.columns
                        scaler_standard = StandardScaler()
                        df_input[features] = scaler_standard.fit_transform(df_input[features])

                        # Make predictions
                        predictions = predict(df_input)
                        if predictions is not None:
                            df["mental_health_condition"] = predictions
                            df['mental_health_condition'] = df['mental_health_condition'].map(MENTAL_HEALTH_CLASSES)

                            # Display results
                            st.subheader("Assessment Results")
                            
                            results_col1, results_col2 = st.columns(2)
                            with results_col1:
                                st.metric("Total Records Processed", len(df))

                            st.write("Detailed Results:")
                            st.dataframe(df[["employee_id", "mental_health_condition"]])

                            # Add download button for results
                            st.download_button(
                                label="Download Results (CSV)",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name="mental_health_assessment_results.csv",
                                mime="text/csv",
                            )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    else:
        # Manual input handling
        st.subheader("Enter Employee Data")
        
        # Create input form with two columns
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
            work_location = st.selectbox("Work Location", ["Onsite", "Hybrid", "Remote"])
            hours_worked = st.number_input("Hours Worked Per Week", min_value=1, max_value=100, value=40)
            meetings = st.number_input("Number of Virtual Meetings", min_value=0, max_value=50, value=5)
            balance = st.slider("Work-Life Balance Rating", 0, 10, 5)
        
        with col2:
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            mental_health = st.selectbox("Access to Mental Health Resources", ["Yes", "No"])
            productivity = st.selectbox("Productivity Change", ["Decrease", "No Change", "Increase"])
            isolation = st.slider("Social Isolation Rating", 0, 10, 5)
            satisfaction = st.selectbox("Satisfaction with Remote Work", ["Unsatisfied", "Neutral", "Satisfied"])
            company_support = st.slider("Company Support for Remote Work", 0, 10, 5)
            physical_activity = st.selectbox("Physical Activity", [0, "Weekly", "Daily"])
            sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])

        if st.button("Assess Mental Health Status"):
            with st.spinner("Analyzing data..."):
                # Create DataFrame from manual input
                df_manual = pd.DataFrame([{
                    "age": age,
                    'gender': gender,
                    "years_of_experience": experience,
                    "work_location": work_location,
                    "hours_worked_per_week": hours_worked,
                    "number_of_virtual_meetings": meetings,
                    "work_life_balance_rating": balance,
                    "stress_level": stress,
                    "access_to_mental_health_resources": mental_health,
                    "productivity_change": productivity,
                    "social_isolation_rating": isolation,
                    "satisfaction_with_remote_work": satisfaction,
                    "company_support_for_remote_work": company_support,
                    "physical_activity": physical_activity,
                    "sleep_quality": sleep_quality
                }])
                
                # Process manual input data
                df_manual = conv_cat_to_numeric(df_manual)

                # Create derived features
                df_manual["age_exp_interaction"] = df_manual["age"] * df_manual["years_of_experience"]
                df_manual["workload"] = df_manual["hours_worked_per_week"] * df_manual["number_of_virtual_meetings"]
                df_manual["sleep_stress_ratio"] = round(df_manual["sleep_quality"] / (df_manual["stress_level"] + 1), 2)
                df_manual["physical_satisfaction"] = df_manual["physical_activity"] * df_manual["satisfaction_with_remote_work"]

                # Make prediction
                prediction = predict(df_manual[model.feature_names_in_])
                if prediction is not None:
                    prediction = prediction[0]
                    
                    # Display results
                    st.subheader("Assessment Results")
                    
                    results_container = st.container()
                    with results_container:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Mental Health Status",
                                MENTAL_HEALTH_CLASSES[prediction],
                                delta=None
                            )

if __name__ == "__main__":
    main()















































