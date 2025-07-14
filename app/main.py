import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="SalaryScope - Advanced AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved contrast and sizing using provided color palette
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #222831;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.4rem;
        color: #112D4E;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        opacity: 1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        letter-spacing: 0.5px;
    }
    .feature-card {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(34, 40, 49, 0.15);
        border: 2px solid #3F72AF;
        min-height: 200px;
        max-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(34, 40, 49, 0.2);
        border-color: #3F72AF;
    }
    .feature-card h3 {
        color: white;
        margin-bottom: 1rem;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .feature-card p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        line-height: 1.6;
        flex-grow: 1;
    }
    .metric-card {
        background: linear-gradient(135deg, #526D82 0%, #3F72AF 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: none;
        box-shadow: 0 4px 15px rgba(63, 114, 175, 0.3);
        min-height: 120px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3F72AF 0%, #112D4E 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(63, 114, 175, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #112D4E 0%, #222831 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(34, 40, 49, 0.4);
    }
    /* Improve info box contrast */
    .stAlert {
        background-color: rgba(82, 109, 130, 0.1);
        color: #222831;
        border: 2px solid #526D82;
        border-radius: 8px;
    }
    .stSuccess {
        background-color: rgba(63, 114, 175, 0.1);
        color: #222831;
        border: 2px solid #3F72AF;
    }
    .stWarning {
        background-color: rgba(57, 62, 70, 0.1);
        color: #222831;
        border: 2px solid #393E46;
    }
    /* Improve sidebar contrast */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #526D82 0%, #393E46 100%);
    }
    section[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #3F72AF 0%, #112D4E 100%);
        color: white;
        width: 100%;
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #222831 0%, #112D4E 100%);
        border-color: rgba(255,255,255,0.4);
    }
    section[data-testid="stSidebar"] h2 {
        color: white;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] h3 {
        color: white;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] p {
        color: rgba(255,255,255,0.9);
    }
    /* Improve form elements - Apply to ALL input types */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    .stSelectbox div[data-baseweb="select"] > div {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stNumberInput > div > div > input,
    .stNumberInput input {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stTextArea > div > div > textarea,
    .stTextArea textarea {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stTextInput > div > div > input,
    .stTextInput input {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stMultiSelect > div > div,
    .stMultiSelect div[data-baseweb="select"] {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stSlider > div > div > div > div,
    .stSlider div[role="slider"] {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
    }
    
    .stDateInput > div > div > input,
    .stDateInput input {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    .stTimeInput > div > div > input,
    .stTimeInput input {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Additional selectors for better coverage */
    input[type="number"],
    input[type="text"],
    input[type="email"],
    input[type="password"],
    textarea,
    select {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        border: 2px solid #3F72AF !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Style placeholder text */
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    input::placeholder,
    textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Style multiselect tags */
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #3F72AF !important;
        color: white !important;
    }
    
    /* Style dropdown options */
    .stSelectbox div[data-baseweb="popover"] div,
    .stMultiSelect div[data-baseweb="popover"] div {
        background: linear-gradient(135deg, #393E46 0%, #526D82 100%) !important;
        color: white !important;
    }
    /* Custom salary display styling */
    .salary-display {
        font-size: 1.5rem;
        font-weight: 700;
        color: #222831;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #3F72AF 0%, #526D82 100%);
        color: white;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(63, 114, 175, 0.3);
    }
    /* Improve metrics display */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #3F72AF 0%, #526D82 100%);
        border: none;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(63, 114, 175, 0.2);
    }
    [data-testid="metric-container"] > div {
        color: white;
    }
    /* Improve dataframe styling */
    .stDataFrame {
        border: 2px solid #526D82;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'skills' not in st.session_state:
    st.session_state.skills = []
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = None

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/trained/salary_prediction_model_xgboost.pkl')
        with open('models/trained/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# High-demand skills list
HIGH_DEMAND_SKILLS = [
    'python', 'machine learning', 'ai', 'artificial intelligence', 'data science',
    'cloud computing', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    'javascript', 'react', 'angular', 'vue', 'node.js', 'java', 'c++',
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'data analysis',
    'tableau', 'power bi', 'excel', 'statistics', 'deep learning',
    'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'computer vision',
    'blockchain', 'cybersecurity', 'devops', 'agile', 'scrum'
]

# Extract skills from text using simple NLP
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in HIGH_DEMAND_SKILLS:
        if skill in text:
            found_skills.append(skill)
    return found_skills

# Calculate skill-based salary adjustment
def calculate_skill_adjustment(skills):
    skill_multipliers = {
        'ai': 1.15, 'artificial intelligence': 1.15, 'machine learning': 1.12,
        'data science': 1.10, 'cloud computing': 1.08, 'aws': 1.08,
        'azure': 1.08, 'gcp': 1.08, 'docker': 1.05, 'kubernetes': 1.06,
        'blockchain': 1.10, 'cybersecurity': 1.09, 'deep learning': 1.12
    }
    
    adjustment = 1.0
    for skill in skills:
        if skill in skill_multipliers:
            adjustment *= skill_multipliers[skill]
    
    return min(adjustment, 1.5)  # Cap at 50% increase

# Generate mock market data
def generate_market_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='ME')
    
    # Generate salary trends for different roles
    roles = ['Data Scientist', 'Software Engineer', 'Product Manager', 'Data Analyst', 'ML Engineer']
    market_data = {}
    
    for role in roles:
        base_salary = np.random.randint(60000, 120000)
        trend = np.random.normal(0.003, 0.002, len(dates))  # 0.3% monthly growth with variation
        salaries = [base_salary]
        
        for t in trend[1:]:
            new_salary = salaries[-1] * (1 + t)
            salaries.append(new_salary)
        
        market_data[role] = {
            'dates': dates,
            'salaries': salaries
        }
    
    return market_data

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = 'home'
    
    if st.button("💰 Predict Salary", use_container_width=True):
        st.session_state.page = 'predict'
    
    if st.button("📊 Education vs Experience", use_container_width=True):
        st.session_state.page = 'education_analysis'
    
    if st.button("📈 Salary Benchmarking", use_container_width=True):
        st.session_state.page = 'benchmarking'
    
    if st.button("🎯 Skills Assessment", use_container_width=True):
        st.session_state.page = 'skills'
    
    if st.button("🚀 Career Progression", use_container_width=True):
        st.session_state.page = 'career_progression'
    
    if st.button("🔄 Role Comparison", use_container_width=True):
        st.session_state.page = 'role_comparison'
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("SalaryScope Advanced uses AI to predict salaries with advanced analytics and personalized insights.")

# Home Page
if st.session_state.page == 'home':
    st.markdown('<h1 class="main-header">Welcome to SalaryScope Advanced 🚀</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Salary Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Advanced Analytics</h3>
            <p>Deep insights into salary trends, market conditions, and career progression paths for informed decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI-Powered Predictions</h3>
            <p>Machine learning models trained on extensive data for accurate and reliable salary predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Personalized Insights</h3>
            <p>Tailored career advice and skill recommendations to maximize your earning potential effectively.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # New features showcase
    st.markdown("## 🌟 New Advanced Features")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.info("📈 **Education vs Experience Analysis** - Visualize salary impact")
        st.info("🎯 **Skills Assessment** - NLP-powered skill analysis")
        st.info("🚀 **Career Progression** - Future salary projections")
    
    with features_col2:
        st.info("💼 **Salary Benchmarking** - Compare with peers")
        st.info("🔄 **Role Comparison** - Cross-role analysis")
        st.info("🤖 **AI Career Advice** - Personalized recommendations")

# Predict Salary Page
elif st.session_state.page == 'predict':
    st.markdown('<h1 class="main-header">Predict Your Salary 💰</h1>', unsafe_allow_html=True)
    
    model, metadata = load_model()
    
    if model is None:
        st.error("Model not loaded. Please ensure model files are present.")
    else:
        with st.form("salary_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=65, value=30)
                experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
                work_hours = st.number_input("Work Hours per Week", min_value=20, max_value=80, value=40)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            with col2:
                education_options = ["Bachelor's", "Master's", "PhD", "High School", "Other"]
                education = st.selectbox("Education Level", education_options)
                
                job_options = ["Software Engineer", "Data Scientist", "Product Manager", "Data Analyst", "Other"]
                job_title = st.selectbox("Job Title", job_options)
                
                industry_options = ["Technology", "Finance", "Healthcare", "Education", "Other"]
                industry = st.selectbox("Industry", industry_options)
            
            # Skills input
            skills_input = st.text_area("List your skills (comma-separated or describe them)", 
                                       placeholder="e.g., Python, Machine Learning, AWS, Data Analysis...")
            
            submitted = st.form_submit_button("Submit", use_container_width=True)
            
            if submitted:
                # Extract skills
                extracted_skills = extract_skills(skills_input)
                st.session_state.skills = extracted_skills
                
                # Prepare input data with correct column order as per model training
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'Education Level': [education],
                    'Job Title': [job_title],
                    'Years of Experience': [experience],
                    'Work Hours': [work_hours],
                    'Industry': [industry]
                })
                
                # Make prediction
                try:
                    predicted_salary = model.predict(input_data)[0]
                    
                    # Apply skill adjustment
                    skill_adjustment = calculate_skill_adjustment(extracted_skills)
                    adjusted_salary = predicted_salary * skill_adjustment
                    
                    st.session_state.predicted_salary = adjusted_salary
                    
                    # Display results
                    st.success("Salary Prediction Complete!")
                    
                    # Display salary range (up to 15K as requested)
                    lower_bound = max(0, adjusted_salary - 7500)
                    upper_bound = adjusted_salary + 7500
                    
                    st.markdown("### 💰 Predicted Salary Range")
                    st.markdown(f'<div class="salary-display">${lower_bound:,.0f} - ${upper_bound:,.0f}</div>', unsafe_allow_html=True)
                    
                    # Skills identified
                    if extracted_skills:
                        st.success(f"🎯 **High-demand skills identified**: {', '.join(extracted_skills)}")
                    
                    # Personalized advice
                    st.markdown("### 🎓 AI-Powered Career Advice")
                    
                    advice_col1, advice_col2 = st.columns(2)
                    
                    with advice_col1:
                        st.markdown("**To increase your salary:**")
                        if 'machine learning' not in extracted_skills and job_title in ['Data Scientist', 'Data Analyst']:
                            st.write("• Learn Machine Learning fundamentals")
                        if 'cloud computing' not in extracted_skills:
                            st.write("• Get certified in cloud platforms (AWS/Azure/GCP)")
                        if education != "Master's" and education != "PhD":
                            st.write("• Consider pursuing advanced education")
                        if experience < 5:
                            st.write("• Focus on gaining more industry experience")
                    
                    with advice_col2:
                        st.markdown("**Recommended certifications:**")
                        if job_title == "Data Scientist":
                            st.write("• AWS Certified Machine Learning")
                            st.write("• Google Cloud Professional Data Engineer")
                        elif job_title == "Software Engineer":
                            st.write("• AWS Solutions Architect")
                            st.write("• Kubernetes Administrator")
                        else:
                            st.write("• PMP Certification")
                            st.write("• Agile/Scrum Master")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

# Education vs Experience Analysis
elif st.session_state.page == 'education_analysis':
    st.markdown('<h1 class="main-header">Education vs Experience Impact Analysis 📊</h1>', unsafe_allow_html=True)
    
    # Generate sample data for visualization
    experience_years = np.arange(0, 21, 1)
    education_levels = ['High School', "Bachelor's", "Master's", 'PhD']
    
    # Create salary data based on education and experience
    salary_data = {}
    base_salaries = {'High School': 35000, "Bachelor's": 50000, "Master's": 65000, 'PhD': 80000}
    experience_multipliers = {'High School': 1500, "Bachelor's": 2500, "Master's": 3500, 'PhD': 4000}
    
    for edu in education_levels:
        salaries = []
        for exp in experience_years:
            salary = base_salaries[edu] + (exp * experience_multipliers[edu]) + (exp ** 1.5 * 500)
            salaries.append(salary)
        salary_data[edu] = salaries
    
    # Create interactive plot
    fig = go.Figure()
    
    colors = {'High School': '#526D82', "Bachelor's": '#3F72AF', "Master's": '#112D4E', 'PhD': '#222831'}
    
    for edu in education_levels:
        fig.add_trace(go.Scatter(
            x=experience_years,
            y=salary_data[edu],
            mode='lines+markers',
            name=edu,
            line=dict(color=colors[edu], width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Salary Progression by Education Level and Experience",
        xaxis_title="Years of Experience",
        yaxis_title="Annual Salary ($)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Key Insights")
        st.write("• **PhD holders** start with the highest base salary")
        st.write("• **Experience impact** is more pronounced for higher education levels")
        st.write("• **10 years experience** can double starting salaries")
        st.write("• **Master's degree** offers best ROI for most careers")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        st.write("• **Early career**: Focus on gaining experience")
        st.write("• **5+ years**: Consider advanced education")
        st.write("• **Industry matters**: Tech values skills over degrees")
        st.write("• **Continuous learning**: Certifications can boost salary")
    
    # ROI Calculator
    st.markdown("### 🧮 Education ROI Calculator")
    
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        current_edu = st.selectbox("Current Education", ["High School", "Bachelor's"])
        target_edu = st.selectbox("Target Education", ["Bachelor's", "Master's", "PhD"])
    
    with calc_col2:
        education_cost = st.number_input("Education Cost ($)", min_value=0, max_value=200000, value=50000)
        years_to_complete = st.number_input("Years to Complete", min_value=1, max_value=6, value=2)
    
    with calc_col3:
        current_exp = st.number_input("Current Experience (years)", min_value=0, max_value=20, value=5)
    
    if st.button("Calculate ROI"):
        # Simple ROI calculation
        current_salary = base_salaries[current_edu] + (current_exp * experience_multipliers[current_edu])
        future_salary = base_salaries[target_edu] + ((current_exp + years_to_complete) * experience_multipliers[target_edu])
        
        salary_increase = future_salary - current_salary
        years_to_break_even = education_cost / salary_increase if salary_increase > 0 else float('inf')
        
        roi_col1, roi_col2, roi_col3 = st.columns(3)
        
        with roi_col1:
            st.metric("Current Salary", f"${current_salary:,.0f}")
        
        with roi_col2:
            st.metric("Projected Salary", f"${future_salary:,.0f}")
        
        with roi_col3:
            st.metric("Break-even Time", f"{years_to_break_even:.1f} years" if years_to_break_even != float('inf') else "N/A")

# Salary Benchmarking
elif st.session_state.page == 'benchmarking':
    st.markdown('<h1 class="main-header">Salary Benchmarking 📈</h1>', unsafe_allow_html=True)
    
    # User input for benchmarking
    bench_col1, bench_col2 = st.columns(2)
    
    with bench_col1:
        job_role = st.selectbox("Select Job Role", 
                               ["Software Engineer", "Data Scientist", "Product Manager", 
                                "Data Analyst", "ML Engineer", "DevOps Engineer"])
        experience_level = st.selectbox("Experience Level", 
                                       ["Entry (0-2 years)", "Mid (3-5 years)", 
                                        "Senior (6-10 years)", "Lead (10+ years)"])
    
    with bench_col2:
        location = st.selectbox("Location", 
                               ["San Francisco", "New York", "Seattle", "Austin", 
                                "Remote", "Other"])
        company_size = st.selectbox("Company Size", 
                                   ["Startup (<50)", "Small (50-200)", 
                                    "Medium (200-1000)", "Large (1000+)"])
    
    if st.button("Generate Benchmark Report"):
        # Generate mock benchmark data
        np.random.seed(42)
        
        # Base salaries by role and experience
        base_salaries = {
            "Software Engineer": {"Entry": 80000, "Mid": 110000, "Senior": 140000, "Lead": 170000},
            "Data Scientist": {"Entry": 90000, "Mid": 120000, "Senior": 150000, "Lead": 180000},
            "Product Manager": {"Entry": 95000, "Mid": 125000, "Senior": 155000, "Lead": 185000},
            "Data Analyst": {"Entry": 65000, "Mid": 85000, "Senior": 105000, "Lead": 125000},
            "ML Engineer": {"Entry": 100000, "Mid": 130000, "Senior": 160000, "Lead": 190000},
            "DevOps Engineer": {"Entry": 85000, "Mid": 115000, "Senior": 145000, "Lead": 175000}
        }
        
        # Location multipliers
        location_mult = {
            "San Francisco": 1.4, "New York": 1.3, "Seattle": 1.2, 
            "Austin": 1.1, "Remote": 1.0, "Other": 0.9
        }
        
        # Get experience key
        exp_key = experience_level.split()[0]
        
        # Calculate salaries
        base_salary = base_salaries[job_role][exp_key]
        location_adjusted = base_salary * location_mult[location]
        
        # Generate distribution
        salaries = np.random.normal(location_adjusted, location_adjusted * 0.15, 1000)
        
        # Create distribution plot
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=salaries,
            nbinsx=30,
            name='Salary Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add percentile lines
        percentiles = [25, 50, 75, 90]
        colors = ['red', 'green', 'blue', 'purple']
        
        for p, color in zip(percentiles, colors):
            value = np.percentile(salaries, p)
            fig.add_vline(x=value, line_dash="dash", line_color=color, 
                         annotation_text=f"{p}th percentile: ${value:,.0f}")
        
        fig.update_layout(
            title=f"Salary Distribution for {job_role} ({experience_level}) in {location}",
            xaxis_title="Annual Salary ($)",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("25th Percentile", f"${np.percentile(salaries, 25):,.0f}")
        
        with stat_col2:
            st.metric("Median (50th)", f"${np.percentile(salaries, 50):,.0f}")
        
        with stat_col3:
            st.metric("75th Percentile", f"${np.percentile(salaries, 75):,.0f}")
        
        with stat_col4:
            st.metric("90th Percentile", f"${np.percentile(salaries, 90):,.0f}")
        
        # Comparison with user's predicted salary
        if st.session_state.predicted_salary:
            user_percentile = (salaries < st.session_state.predicted_salary).sum() / len(salaries) * 100
            
            st.info(f"📊 Your predicted salary of ${st.session_state.predicted_salary:,.0f} is at the "
                   f"**{user_percentile:.0f}th percentile** for this role and location.")
            
            if user_percentile < 50:
                st.warning("💡 Consider negotiating for a higher salary or gaining more skills to increase your market value.")
            else:
                st.success("✅ Your salary is competitive for your role and location!")

# Skills Assessment
elif st.session_state.page == 'skills':
    st.markdown('<h1 class="main-header">Skills Assessment & Analysis 🎯</h1>', unsafe_allow_html=True)
    
    st.markdown("### Enter Your Skills")
    skills_text = st.text_area("Describe your skills, technologies, and expertise", 
                              height=150,
                              placeholder="Example: I have 5 years of experience in Python programming, "
                                        "machine learning with TensorFlow and scikit-learn, "
                                        "cloud computing with AWS, and data visualization with Tableau...")
    
    if st.button("Analyze Skills"):
        if skills_text:
            # Extract skills
            extracted_skills = extract_skills(skills_text)
            
            if extracted_skills:
                st.success(f"🎯 Identified {len(extracted_skills)} relevant skills!")
                
                # Display skills with categories
                skill_categories = {
                    'Programming': ['python', 'javascript', 'java', 'c++', 'react', 'angular', 'vue', 'node.js'],
                    'Data & AI': ['machine learning', 'ai', 'artificial intelligence', 'data science', 
                                 'data analysis', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 
                                 'nlp', 'computer vision'],
                    'Cloud & DevOps': ['cloud computing', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'devops'],
                    'Databases': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql'],
                    'Analytics': ['tableau', 'power bi', 'excel', 'statistics'],
                    'Other': ['blockchain', 'cybersecurity', 'agile', 'scrum']
                }
                
                categorized_skills = {}
                for category, skills_list in skill_categories.items():
                    found = [s for s in extracted_skills if s in skills_list]
                    if found:
                        categorized_skills[category] = found
                
                # Display categorized skills
                cols = st.columns(3)
                for i, (category, skills) in enumerate(categorized_skills.items()):
                    with cols[i % 3]:
                        st.markdown(f"**{category}**")
                        for skill in skills:
                            st.write(f"• {skill.title()}")
                
                # Skill demand analysis
                st.markdown("### 📊 Skill Demand Analysis")
                
                # Create skill demand chart
                skill_demand = {
                    'python': 95, 'machine learning': 92, 'ai': 90, 'cloud computing': 88,
                    'aws': 87, 'javascript': 85, 'data science': 85, 'docker': 82,
                    'kubernetes': 80, 'sql': 78, 'react': 75, 'tensorflow': 73,
                    'azure': 72, 'gcp': 70, 'data analysis': 68, 'tableau': 65
                }
                
                user_skill_demand = {s: skill_demand.get(s, 50) for s in extracted_skills if s in skill_demand}
                
                if user_skill_demand:
                    fig = go.Figure(data=[
                        go.Bar(x=list(user_skill_demand.keys()), 
                              y=list(user_skill_demand.values()),
                              marker_color='lightgreen')
                    ])
                    
                    fig.update_layout(
                        title="Market Demand for Your Skills",
                        xaxis_title="Skills",
                        yaxis_title="Demand Score (0-100)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Salary impact
                skill_adjustment = calculate_skill_adjustment(extracted_skills)
                st.metric("Estimated Salary Impact", f"+{(skill_adjustment-1)*100:.1f}%")
                
                # Skill recommendations
                st.markdown("### 💡 Recommended Skills to Learn")
                
                missing_high_demand = [s for s in ['machine learning', 'cloud computing', 'docker', 'kubernetes'] 
                                     if s not in extracted_skills]
                
                if missing_high_demand:
                    rec_cols = st.columns(len(missing_high_demand))
                    for i, skill in enumerate(missing_high_demand):
                        with rec_cols[i]:
                            st.info(f"**{skill.title()}**\n\nHigh demand skill that could increase your salary by 5-15%")
            else:
                st.warning("No specific technical skills identified. Try being more specific about technologies and tools.")
        else:
            st.warning("Please enter your skills to analyze.")

# Career Progression
elif st.session_state.page == 'career_progression':
    st.markdown('<h1 class="main-header">Career Progression & Future Salary 🚀</h1>', unsafe_allow_html=True)
    
    # Input for career progression - Better aligned layout
    st.markdown("### 📝 Career Information")
    
    # First row - Role and Salary
    prog_col1, prog_col2 = st.columns(2)
    
    with prog_col1:
        current_role = st.selectbox("Current Role", 
                                   ["Junior Developer", "Software Engineer", "Senior Engineer", 
                                    "Data Analyst", "Data Scientist", "Product Manager"])
    
    with prog_col2:
        current_salary = st.number_input("Current Salary ($)", 
                                       min_value=30000, max_value=300000, value=80000, step=5000)
    
    # Second row - Years and Path
    prog_col3, prog_col4 = st.columns(2)
    
    with prog_col3:
        years_ahead = st.slider("Years to Project", min_value=1, max_value=10, value=5)
    
    with prog_col4:
        career_path = st.selectbox("Career Path",
                                  ["Technical Track", "Management Track", "Specialist Track"])
    
    st.markdown("---")
    
    if st.button("Generate Career Projection"):
        # Define career progression paths
        progression_paths = {
            "Technical Track": {
                "Junior Developer": ["Software Engineer", "Senior Engineer", "Staff Engineer", "Principal Engineer"],
                "Software Engineer": ["Senior Engineer", "Staff Engineer", "Principal Engineer", "Distinguished Engineer"],
                "Data Analyst": ["Senior Analyst", "Data Scientist", "Senior Data Scientist", "Principal Data Scientist"]
            },
            "Management Track": {
                "Junior Developer": ["Software Engineer", "Team Lead", "Engineering Manager", "Director of Engineering"],
                "Software Engineer": ["Senior Engineer", "Team Lead", "Engineering Manager", "VP of Engineering"],
                "Data Analyst": ["Senior Analyst", "Analytics Lead", "Analytics Manager", "Director of Analytics"]
            },
            "Specialist Track": {
                "Junior Developer": ["Software Engineer", "Domain Expert", "Technical Specialist", "Chief Architect"],
                "Software Engineer": ["Senior Engineer", "Technical Expert", "Solutions Architect", "Chief Technology Officer"],
                "Data Analyst": ["Senior Analyst", "Analytics Expert", "Data Architect", "Chief Data Officer"]
            }
        }
        
        # Generate salary progression
        years = list(range(years_ahead + 1))
        
        # Different growth rates for different paths
        growth_rates = {
            "Technical Track": 0.08,  # 8% annual growth
            "Management Track": 0.12,  # 12% annual growth
            "Specialist Track": 0.10   # 10% annual growth
        }
        
        base_growth = growth_rates[career_path]
        salaries = [current_salary]
        roles = [current_role]
        
        # Get progression path
        if current_role in progression_paths[career_path]:
            role_progression = progression_paths[career_path][current_role]
        else:
            role_progression = ["Senior " + current_role, "Lead " + current_role,
                              "Principal " + current_role, "Executive"]
        
        for year in range(1, years_ahead + 1):
            # Calculate salary with some randomness
            growth = base_growth + np.random.normal(0, 0.02)
            new_salary = salaries[-1] * (1 + growth)
            
            # Add promotion bonuses
            if year % 3 == 0 and len(role_progression) > year // 3 - 1:
                new_salary *= 1.15  # 15% promotion bonus
                new_role = role_progression[min(year // 3 - 1, len(role_progression) - 1)]
                roles.append(new_role)
            else:
                roles.append(roles[-1])
            
            salaries.append(new_salary)
        
        # Create progression chart
        fig = go.Figure()
        
        # Add salary line
        fig.add_trace(go.Scatter(
            x=years,
            y=salaries,
            mode='lines+markers',
            name='Projected Salary',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ))
        
        # Add role annotations
        for i, (year, role) in enumerate(zip(years, roles)):
            if i == 0 or roles[i] != roles[i-1]:
                fig.add_annotation(
                    x=year,
                    y=salaries[i],
                    text=role,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    ax=0,
                    ay=-40
                )
        
        fig.update_layout(
            title=f"Career Progression: {career_path}",
            xaxis_title="Years from Now",
            yaxis_title="Annual Salary ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        met_col1, met_col2, met_col3 = st.columns(3)
        
        with met_col1:
            st.metric("Starting Salary", f"${current_salary:,.0f}")
        
        with met_col2:
            st.metric(f"Salary in {years_ahead} Years", f"${salaries[-1]:,.0f}")
        
        with met_col3:
            total_growth = ((salaries[-1] - current_salary) / current_salary) * 100
            st.metric("Total Growth", f"{total_growth:.1f}%")
        
        # Career milestones
        st.markdown("### 🎯 Projected Career Milestones")
        
        milestone_data = []
        for i, (year, role, salary) in enumerate(zip(years, roles, salaries)):
            if i == 0 or roles[i] != roles[i-1]:
                milestone_data.append({
                    "Year": year,
                    "Role": role,
                    "Salary": f"${salary:,.0f}",
                    "Growth": f"+{((salary - current_salary) / current_salary * 100):.1f}%" if year > 0 else "Current"
                })
        
        st.dataframe(pd.DataFrame(milestone_data), use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations to Accelerate Career Growth")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("**Skills to Develop:**")
            if career_path == "Technical Track":
                st.write("• Deep technical expertise")
                st.write("• System design & architecture")
                st.write("• Open source contributions")
                st.write("• Technical mentoring")
            elif career_path == "Management Track":
                st.write("• Leadership & people management")
                st.write("• Strategic planning")
                st.write("• Budget management")
                st.write("• Stakeholder communication")
            else:
                st.write("• Domain expertise")
                st.write("• Industry certifications")
                st.write("• Thought leadership")
                st.write("• Consulting skills")
        
        with rec_col2:
            st.markdown("**Action Items:**")
            st.write("• Set quarterly skill development goals")
            st.write("• Seek mentorship from senior professionals")
            st.write("• Take on high-visibility projects")
            st.write("• Build your professional network")
            st.write("• Consider advanced education or certifications")

# Role Comparison
elif st.session_state.page == 'role_comparison':
    st.markdown('<h1 class="main-header">Job Role Comparison Tool 🔄</h1>', unsafe_allow_html=True)
    
    st.markdown("Compare salary predictions across different job roles to make informed career decisions.")
    
    # Input section
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        base_experience = st.number_input("Years of Experience", min_value=0, max_value=30, value=5)
        base_education = st.selectbox("Education Level",
                                    ["High School", "Bachelor's", "Master's", "PhD"])
    
    with comp_col2:
        base_location = st.selectbox("Location",
                                   ["San Francisco", "New York", "Seattle", "Austin", "Remote"])
        base_skills = st.multiselect("Select Your Skills",
                                   ['Python', 'Machine Learning', 'Cloud Computing', 'JavaScript',
                                    'Data Analysis', 'SQL', 'Docker', 'Kubernetes'])
    
    # Roles to compare
    roles_to_compare = st.multiselect("Select Roles to Compare (up to 5)",
                                     ["Software Engineer", "Data Scientist", "Product Manager",
                                      "Data Analyst", "ML Engineer", "DevOps Engineer",
                                      "Full Stack Developer", "Backend Engineer", "Frontend Engineer"],
                                     default=["Software Engineer", "Data Scientist", "Product Manager"])
    
    if st.button("Compare Roles") and roles_to_compare:
        # Generate comparison data
        comparison_data = []
        
        # Base salaries and factors (market-standardized)
        role_base_salaries = {
            "Software Engineer": 65000,
            "Data Scientist": 75000,
            "Product Manager": 80000,
            "Data Analyst": 55000,
            "ML Engineer": 85000,
            "DevOps Engineer": 70000,
            "Full Stack Developer": 60000,
            "Backend Engineer": 65000,
            "Frontend Engineer": 58000
        }
        
        education_multipliers = {
            "High School": 0.8,
            "Bachelor's": 1.0,
            "Master's": 1.15,
            "PhD": 1.25
        }
        
        location_multipliers = {
            "San Francisco": 1.4,
            "New York": 1.3,
            "Seattle": 1.2,
            "Austin": 1.1,
            "Remote": 1.0
        }
        
        for role in roles_to_compare:
            base_salary = role_base_salaries.get(role, 85000)
            
            # Apply multipliers
            salary = base_salary
            salary *= (1 + base_experience * 0.03)  # 3% per year of experience
            salary *= education_multipliers[base_education]
            salary *= location_multipliers[base_location]
            
            # Skill adjustments
            skill_bonus = len(base_skills) * 0.02  # 2% per skill
            salary *= (1 + skill_bonus)
            
            # Calculate growth potential
            growth_potential = "High" if role in ["ML Engineer", "Data Scientist", "Product Manager"] else "Medium"
            
            comparison_data.append({
                "Role": role,
                "Predicted Salary": salary,
                "Growth Potential": growth_potential,
                "Market Demand": np.random.choice(["Very High", "High", "Medium"], p=[0.4, 0.4, 0.2])
            })
        
        # Sort by salary
        comparison_data.sort(key=lambda x: x["Predicted Salary"], reverse=True)
        
        # Create comparison chart
        fig = go.Figure()
        
        roles = [d["Role"] for d in comparison_data]
        salaries = [d["Predicted Salary"] for d in comparison_data]
        
        colors = ['green' if d["Growth Potential"] == "High" else 'blue' for d in comparison_data]
        
        fig.add_trace(go.Bar(
            x=roles,
            y=salaries,
            marker_color=colors,
            text=[f"${s:,.0f}" for s in salaries],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Salary Comparison Across Roles",
            xaxis_title="Job Role",
            yaxis_title="Predicted Salary ($)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("### 📊 Detailed Comparison")
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison["Predicted Salary"] = df_comparison["Predicted Salary"].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(df_comparison, use_container_width=True)
        
        # Role insights
        st.markdown("### 💡 Role-Specific Insights")
        
        for i, role_data in enumerate(comparison_data[:3]):  # Top 3 roles
            role = role_data["Role"]
            
            with st.expander(f"{i+1}. {role}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Responsibilities:**")
                    if role == "Software Engineer":
                        st.write("• Design and develop software applications")
                        st.write("• Write clean, maintainable code")
                        st.write("• Collaborate with cross-functional teams")
                    elif role == "Data Scientist":
                        st.write("• Analyze complex data sets")
                        st.write("• Build predictive models")
                        st.write("• Communicate insights to stakeholders")
                    elif role == "Product Manager":
                        st.write("• Define product strategy and roadmap")
                        st.write("• Work with engineering and design teams")
                        st.write("• Analyze market and user needs")
                    else:
                        st.write("• Role-specific technical tasks")
                        st.write("• Team collaboration")
                        st.write("• Continuous learning and improvement")
                
                with col2:
                    st.markdown("**Required Skills:**")
                    if role == "Software Engineer":
                        st.write("• Programming languages (Python, Java, etc.)")
                        st.write("• Software design patterns")
                        st.write("• Version control (Git)")
                    elif role == "Data Scientist":
                        st.write("• Statistical analysis")
                        st.write("• Machine learning")
                        st.write("• Data visualization")
                    elif role == "Product Manager":
                        st.write("• Strategic thinking")
                        st.write("• Data analysis")
                        st.write("• Communication skills")
                    else:
                        st.write("• Technical expertise")
                        st.write("• Problem-solving")
                        st.write("• Communication")
        
        # Transition advice
        st.markdown("### 🚀 Career Transition Advice")
        
        current_role = st.selectbox("What's your current role?",
                                   ["None"] + list(role_base_salaries.keys()))
        
        if current_role != "None" and current_role not in roles_to_compare:
            target_role = roles_to_compare[0]  # Highest paying from comparison
            
            st.info(f"**Transitioning from {current_role} to {target_role}:**")
            
            trans_col1, trans_col2 = st.columns(2)
            
            with trans_col1:
                st.markdown("**Skills to Develop:**")
                if target_role == "Data Scientist" and current_role == "Software Engineer":
                    st.write("• Statistics and probability")
                    st.write("• Machine learning algorithms")
                    st.write("• Data visualization tools")
                elif target_role == "Product Manager":
                    st.write("• Business strategy")
                    st.write("• User research methods")
                    st.write("��� Product analytics")
                else:
                    st.write("• Role-specific technical skills")
                    st.write("• Industry knowledge")
                    st.write("• Soft skills development")
            
            with trans_col2:
                st.markdown("**Recommended Steps:**")
                st.write("• Take online courses or certifications")
                st.write("• Work on side projects")
                st.write("• Network with professionals in target role")
                st.write("• Seek mentorship")
                st.write("• Consider internal transfers")

# Footer
st.markdown("---")
st.markdown("### 📝 Disclaimer")
st.markdown("Salary predictions are estimates based on market data and should be used as guidance only. "
           "Actual salaries may vary based on company, location, and individual negotiations.")

# Dynamic Market Conditions Note
st.markdown("### 🌐 Market Conditions")
st.info("Note: This app uses simulated market data for demonstration. In a production environment, "
        "real-time data from APIs would be integrated for dynamic market adjustments.")