# SalaryScope - Advanced AI Salary Predictor 💰

An advanced AI-powered salary prediction platform built with Streamlit, featuring comprehensive analytics, career progression insights, and personalized recommendations.

## 🚀 Features

- **AI-Powered Salary Predictions**: Machine learning models for accurate salary estimation
- **Advanced Analytics**: Deep insights into salary trends and market conditions
- **Career Progression**: Future salary projections and career path analysis
- **Skills Assessment**: NLP-powered skill analysis and recommendations
- **Salary Benchmarking**: Compare with industry peers and market standards
- **Role Comparison**: Cross-role analysis for career decision making
- **Interactive Visualizations**: Rich charts and graphs using Plotly

## 📁 Project Structure

```
SalaryScope/
├── app/
│   ├── __init__.py
│   └── main.py                 # Main Streamlit application
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── DataSet1.csv
│   │   ├── DataSet2.csv
│   │   └── DataSet3.csv
│   └── processed/              # Cleaned and processed data
│       └── cleaned_salary_data.csv
├── models/
│   └── trained/                # Trained model files
│       ├── salary_prediction_model_xgboost.pkl
│       └── model_metadata.json
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
├── assets/
│   ├── images/                 # Charts and visualizations
│   │   ├── Correlation matrix for numerical features.png
│   │   ├── model performance.png
│   │   ├── Prediction vs Actual.png
│   │   ├── Salary by categorical features.png
│   │   ├── salary distribution.png
│   │   └── Top 20 features.png
│   └── styles/                 # CSS and styling files
│       └── custom.css
├── requirements.txt
├── .gitignore
└── README.md
```

## 📋 Dependencies

The project requires the following main dependencies:

- **streamlit==1.28.1** - Web application framework
- **pandas==2.1.3** - Data manipulation and analysis
- **numpy==1.26.2** - Numerical computing
- **scikit-learn==1.3.2** - Machine learning library
- **xgboost==2.0.2** - Gradient boosting framework
- **joblib==1.3.2** - Model serialization
- **plotly==5.18.0** - Interactive visualizations

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SalaryScope.git
   cd SalaryScope
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

## 🚀 Quick Start

1. Navigate to the application URL (typically `http://localhost:8501`)
2. Use the sidebar to navigate between different features
3. Start with "Predict Salary" to get your salary estimation
4. Explore other features like Skills Assessment and Career Progression

## 📊 Model Information

- **Algorithm**: XGBoost Regressor
- **Features**: Age, Experience, Education, Job Title, Industry, Work Hours, Skills
- **Performance**: High accuracy with comprehensive feature engineering
- **Data Sources**: Multiple salary datasets with 10,000+ records

## 📈 Available Visualizations

The application includes comprehensive data visualizations:

- **Correlation Matrix** - Shows relationships between numerical features
- **Model Performance** - Displays model accuracy and evaluation metrics
- **Prediction vs Actual** - Compares predicted salaries with actual values
- **Salary Distribution** - Visualizes salary distribution across the dataset
- **Salary by Features** - Shows salary variations by categorical features
- **Top 20 Features** - Highlights the most important features for prediction

## 🎨 Color Palette

The application uses a professional color scheme:
- Primary: `#222831` (Dark Gray)
- Secondary: `#393E46` (Medium Gray)
- Accent: `#3F72AF` (Blue)
- Dark Accent: `#112D4E` (Dark Blue)
- Light Accent: `#526D82` (Light Blue-Gray)

## 🔧 Configuration

The application configuration is managed through:
- Model parameters defined in the main application
- UI settings configured in Streamlit
- Data processing options in the notebooks
- Custom styling in `assets/styles/custom.css`

## 📈 Features Overview

### 1. Salary Prediction
- Input personal and professional details
- Get AI-powered salary range predictions
- Skill-based adjustments and recommendations

### 2. Education vs Experience Analysis
- Visualize salary impact of education levels
- ROI calculator for education investments
- Experience progression insights

### 3. Salary Benchmarking
- Compare with industry standards
- Location-based adjustments
- Percentile rankings

### 4. Skills Assessment
- NLP-powered skill extraction
- Market demand analysis
- Skill gap identification

### 5. Career Progression
- Future salary projections
- Career path recommendations
- Growth rate analysis

### 6. Role Comparison
- Cross-role salary analysis
- Career transition advice
- Market demand insights

## 🧪 Testing

Run tests using:
```bash
python -m pytest tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

For support, email support@salaryscope.com or create an issue in the repository.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- XGBoost developers for the powerful ML library
- Plotly for interactive visualizations
- Open source community for various tools and libraries

---

**Built with ❤️ using Streamlit, XGBoost, and modern web technologies**