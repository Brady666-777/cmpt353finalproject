# VancouverPy: A Data-Driven Framework for Predicting Restaurant Success

A comprehensive data science project that predicts optimal locations for new restaurants in Vancouver, BC, using machine learning and multiple data sources.

## Project Overview

This project aims to transform the subjective art of restaurant site selection into a data-driven science by:

- Integrating diverse datasets (business licenses, demographics, Yelp reviews, transit data)
- Engineering a quantitative "Success Score" for existing establishments
- Training machine learning models to predict success scores for new locations
- Generating predictive heat maps across Vancouver

## Data Sources

- **City of Vancouver Open Data Portal**: Business licenses, local area boundaries, traffic counts
- **Statistics Canada**: 2021 Census profiles with demographics and income data
- **Yelp Fusion API**: Restaurant reviews, ratings, and performance metrics

## Key Features

- **Multi-source Data Integration**: Combines official municipal data with crowd-sourced performance metrics
- **Advanced Feature Engineering**: Creates meaningful predictors like competitive density and affordability mismatch
- **Machine Learning Pipeline**: Tests multiple algorithms (Random Forest, XGBoost, etc.) for optimal performance
- **Geospatial Analysis**: Leverages location-based insights and spatial relationships
- **Interactive Visualizations**: Generates heat maps and geographic visualizations

## Project Structure

```
cmpt353finalproject/
├── src/                          # Python scripts for data processing
│   ├── 01_get_data.py           # Data collection from APIs
│   ├── 02_clean_and_feature_engineer.py  # Data cleaning and feature engineering
│   └── 03_model_training.py     # Machine learning model training
├── data/                        # Data storage
│   ├── raw/                     # Raw datasets from sources
│   └── processed/               # Cleaned and engineered features
├── models/                      # Trained models and artifacts
├── notebooks/                   # Jupyter notebooks for analysis
│   └── model_training.ipynb     # Interactive model training and evaluation
├── reports/                     # Final reports and documentation
│   └── plots/                   # Generated visualizations
├── .github/                     # GitHub configuration
│   └── copilot-instructions.md  # AI assistant instructions
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── launcher.py                  # Project launcher script
└── README.md                    # This file
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment manager (venv, conda, etc.)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/cmpt353finalproject.git
   cd cmpt353finalproject
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**
   ```bash
   cp .env.example .env
   # Edit .env file with your actual API keys
   ```

### Required API Keys

1. **City of Vancouver API Key**

   - Register at: [Vancouver Open Data Portal](https://opendata.vancouver.ca/)
   - Add to `.env` file as `VANCOUVER_API_KEY`

2. **Yelp Fusion API Key**
   - Register at: [Yelp Developers](https://www.yelp.com/developers/v3/manage_app)
   - Add to `.env` file as `YELP_API_KEY`

## Usage Instructions

### 1. Data Collection

```bash
cd src
python 01_get_data.py
```

This script will collect data from all configured sources and save raw datasets to `data/raw/`.

### 2. Data Processing

```bash
python 02_clean_and_feature_engineer.py
```

This script cleans the raw data, engineers features, and saves processed datasets to `data/processed/`.

### 3. Model Training (Python Script)

```bash
python 03_model_training.py
```

This script runs the complete machine learning pipeline:

- Exploratory data analysis
- Feature importance analysis
- Multiple model training and comparison
- Best model evaluation and interpretation
- Prediction heat map generation
- Model saving for deployment

### 4. Interactive Analysis (Optional)

```bash
jupyter notebook notebooks/model_training.ipynb
```

Open the Jupyter notebook for interactive analysis and custom experimentation.

## Methodology

### Feature Engineering

- **Competitive Landscape**: Density of competitors and complementary businesses
- **Neighborhood Profiling**: Demographics, income, and cuisine diversity
- **Transit Accessibility**: Distance to stations and bus stop density
- **Affordability Mismatch**: Alignment between price point and local income

### Success Score Creation

Composite metric combining:

- Yelp rating (40% weight)
- Review count (40% weight)
- Operational longevity (20% weight)

### Machine Learning Models

- Linear Regression & Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Expected Outputs

1. **Processed Datasets**: Clean, geocoded restaurant data with engineered features
2. **Trained Models**: Multiple ML models saved in `models/` directory
3. **Model Comparison**: Performance metrics and cross-validation results
4. **Feature Importance**: Analysis of key factors driving restaurant success
5. **Visualizations**: Comprehensive plots saved in `reports/plots/`
6. **Prediction Heat Maps**: Geographic visualizations of success potential
7. **Model Artifacts**: Saved models ready for deployment and prediction

## Sample Results

_Note: Results will be populated after running the analysis with real data_

- Model Performance: R² score, RMSE, MAE
- Top Success Factors: Most important features identified
- Geographic Insights: Neighborhoods with highest success potential

## Future Enhancements

- [ ] Real-time data integration
- [ ] Web application for interactive predictions
- [ ] Advanced deep learning models
- [ ] Integration with business outcome validation
- [ ] Expansion to other Canadian cities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- City of Vancouver for providing open data access
- Yelp for restaurant performance data
- Statistics Canada for demographic insights
- CMPT 353 course instructors and teaching assistants

## Contact

- **Project Repository**: [GitHub Link](https://github.com/your-username/cmpt353finalproject)
- **Course**: CMPT 353 - Computational Data Science
- **Institution**: Simon Fraser University

---

**Disclaimer**: This project is for educational purposes. Actual business decisions should consider additional factors beyond the model's predictions.
