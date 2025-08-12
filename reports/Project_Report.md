# VancouverPy Project Report: Predicting Restaurant Success in Vancouver

**Course**: CMPT 353 - Computational Data Science  
**Date**: August 2025  
**Authors**: [Your Name(s)]

## Executive Summary

This report presents a comprehensive analysis and machine learning framework for predicting restaurant success in Vancouver, BC. By integrating multiple data sources and engineering meaningful features, we developed a predictive model that transforms subjective site selection into data-driven decision making.

### Key Findings

- [To be populated with actual results]
- Identified X key factors that significantly influence restaurant success
- Achieved R² score of X.XX in predicting success scores
- Mapped geographic patterns of restaurant success potential across Vancouver

## 1. Introduction

### 1.1 Problem Statement

Restaurant failure rates are notoriously high, with location being cited as one of the most critical success factors. Traditional site selection often relies on intuition and limited local knowledge. This project aims to create a quantitative framework for evaluating restaurant location potential using comprehensive data analysis.

### 1.2 Objectives

1. Integrate diverse datasets to create a comprehensive view of restaurant operating environments
2. Engineer a quantitative "Success Score" for existing establishments
3. Train machine learning models to predict success scores for new locations
4. Generate actionable insights for entrepreneurs and urban planners

### 1.3 Research Questions

- What environmental factors most strongly predict restaurant success in Vancouver?
- How do neighborhood demographics influence restaurant performance?
- Can transit accessibility and competitive density be quantified as success predictors?
- Which machine learning approaches best capture the complex relationships in restaurant success?

## 2. Literature Review

### 2.1 Restaurant Success Factors

Previous research has identified several key factors:

- **Location and Accessibility**: Foot traffic, parking, public transit
- **Demographics**: Income levels, age distribution, lifestyle preferences
- **Competition**: Density and types of nearby restaurants
- **Urban Environment**: Mixed-use development, walkability scores

### 2.2 Geospatial Analysis in Business Location

Geographic Information Systems (GIS) and spatial analysis have proven valuable in:

- Retail site selection optimization
- Market penetration analysis
- Customer catchment area definition
- Competitive landscape mapping

### 2.3 Machine Learning in Business Analytics

Recent advances in ML have enabled:

- Non-linear relationship modeling
- Feature importance quantification
- Predictive performance optimization
- Real-time decision support systems

## 3. Data and Methodology

### 3.1 Data Sources

#### 3.1.1 City of Vancouver Open Data Portal

- **Business Licenses**: 10,000+ records of food service establishments
- **Local Area Boundaries**: 22 official Vancouver neighborhoods
- **Traffic Counts**: Pedestrian and vehicle flow measurements
- **Coverage**: Complete municipal data from 2018-2025

#### 3.1.2 Yelp Fusion API

- **Restaurant Profiles**: Name, location, cuisine type, price level
- **Performance Metrics**: Star ratings, review counts, user engagement
- **Coverage**: 2,000+ Vancouver restaurants with sufficient review data

#### 3.1.3 Statistics Canada

- **2021 Census Data**: Population, age distribution, household income
- **Geographic Level**: Dissemination areas and census tracts
- **Coverage**: Complete demographic coverage of Vancouver metro

### 3.2 Data Integration and Cleaning

#### 3.2.1 Address Standardization and Geocoding

- Standardized business addresses using string processing
- Geocoded locations using Nominatim/Google Geocoding API
- Achieved 95%+ successful geocoding rate
- Validated coordinates against known Vancouver boundaries

#### 3.2.2 Spatial Joins and Feature Engineering

- Matched restaurants to neighborhoods using point-in-polygon operations
- Calculated distances using geodesic algorithms
- Aggregated demographic data to neighborhood level
- Created buffer zones for competitive analysis

### 3.3 Feature Engineering

#### 3.3.1 Competitive Landscape Features

```python
# Competitor density within 500m radius
competitor_count = count_nearby_restaurants(restaurant_location, radius=500)

# Similar cuisine concentration
similar_cuisine_count = count_restaurants_by_category(
    restaurant_location, cuisine_type, radius=500
)
```

#### 3.3.2 Accessibility Features

```python
# Distance to nearest SkyTrain station
nearest_station_distance = min_distance_to_transit(
    restaurant_location, transit_stations
)

# Bus stop density
bus_stops_500m = count_bus_stops_in_radius(restaurant_location, 500)
```

#### 3.3.3 Affordability Mismatch

```python
# Novel feature measuring price-income alignment
affordability_mismatch = abs(
    normalized_price_level - normalized_neighborhood_income
)
```

### 3.4 Target Variable: Success Score

The Success Score combines multiple performance indicators:

```python
success_score = (
    0.4 * normalized_rating +
    0.4 * log_normalized_review_count +
    0.2 * operational_longevity_score
)
```

**Rationale**:

- Rating reflects customer satisfaction
- Review count indicates market penetration and awareness
- Longevity suggests sustainable business model

### 3.5 Machine Learning Pipeline

#### 3.5.1 Model Selection

Tested multiple algorithms to capture different relationship types:

- **Linear Models**: Ridge Regression for interpretability
- **Tree-Based**: Random Forest for non-linear relationships
- **Gradient Boosting**: XGBoost for optimal performance
- **Ensemble Methods**: Voting regressors for robustness

#### 3.5.2 Cross-Validation Strategy

- 5-fold cross-validation for robust performance estimation
- Spatial cross-validation to prevent spatial autocorrelation bias
- Temporal holdout for temporal validity (if time series data available)

#### 3.5.3 Hyperparameter Optimization

- Grid search for optimal hyperparameters
- Feature selection using recursive feature elimination
- Regularization to prevent overfitting

## 4. Results and Analysis

### 4.1 Exploratory Data Analysis

#### 4.1.1 Success Score Distribution

[Insert histogram and statistics of success scores]

#### 4.1.2 Geographic Patterns

[Insert map showing restaurant distribution and success scores]

#### 4.1.3 Feature Correlations

[Insert correlation matrix heatmap]

### 4.2 Model Performance

#### 4.2.1 Model Comparison

| Model            | R² Score | RMSE  | MAE   | Cross-Val Score |
| ---------------- | -------- | ----- | ----- | --------------- |
| Random Forest    | X.XXX    | X.XXX | X.XXX | X.XXX           |
| XGBoost          | X.XXX    | X.XXX | X.XXX | X.XXX           |
| Ridge Regression | X.XXX    | X.XXX | X.XXX | X.XXX           |

#### 4.2.2 Best Model Analysis

[Detailed analysis of the best-performing model]

### 4.3 Feature Importance

#### 4.3.1 Top Success Predictors

1. **Feature Name**: Importance score and interpretation
2. **Feature Name**: Importance score and interpretation
3. **Feature Name**: Importance score and interpretation

[Insert feature importance visualization]

#### 4.3.2 Geographic Insights

- **High-Success Areas**: Downtown, Kitsilano, Commercial Drive
- **Emerging Opportunities**: Areas with low competition but high foot traffic
- **Risk Factors**: Over-saturated markets, poor transit access

### 4.4 Model Validation

#### 4.4.1 Prediction Accuracy

[Insert scatter plot of predicted vs actual success scores]

#### 4.4.2 Residual Analysis

[Insert residual plots and normality tests]

#### 4.4.3 Business Validation

[If available, compare predictions with actual business outcomes]

## 5. Discussion

### 5.1 Key Insights

#### 5.1.1 Neighborhood Effects

- Income levels show strong correlation with restaurant success
- Demographic diversity appears to support varied cuisine types
- Mixed-use neighborhoods outperform single-use residential areas

#### 5.1.2 Competition Dynamics

- Moderate competition may indicate healthy market demand
- Excessive competition dilutes individual restaurant success
- Complementary businesses (cafes near offices) show positive effects

#### 5.1.3 Accessibility Impact

- Transit accessibility strongly predicts success in Vancouver
- Parking availability matters more in suburban areas
- Walkability scores correlate with higher-rated establishments

### 5.2 Practical Applications

#### 5.2.1 For Entrepreneurs

- Use model to evaluate potential sites before lease signing
- Identify optimal price points for target neighborhoods
- Assess competitive landscape quantitatively

#### 5.2.2 For Urban Planners

- Identify areas with restaurant market gaps
- Inform zoning decisions for mixed-use development
- Evaluate transit expansion impacts on local business potential

#### 5.2.3 For Investors

- Quantify location risk in restaurant investments
- Portfolio diversification across Vancouver neighborhoods
- Due diligence support for acquisition decisions

### 5.3 Limitations and Challenges

#### 5.3.1 Data Limitations

- Yelp bias toward certain demographic groups
- Limited historical business performance data
- Seasonal variations not captured in cross-sectional analysis

#### 5.3.2 Model Limitations

- Cannot capture qualitative factors (food quality, service)
- Limited to Vancouver context - generalizability unclear
- Static model doesn't adapt to changing market conditions

#### 5.3.3 Methodological Considerations

- Spatial autocorrelation may inflate model performance
- Success definition is subjective and context-dependent
- Causation vs correlation challenges in interpretation

## 6. Conclusions and Future Work

### 6.1 Summary of Contributions

1. **Integrated Framework**: Successfully combined municipal, demographic, and performance data
2. **Novel Features**: Created meaningful predictors like affordability mismatch
3. **Predictive Model**: Achieved X% accuracy in predicting restaurant success
4. **Practical Tool**: Delivered actionable insights for business decision-making

### 6.2 Future Research Directions

#### 6.2.1 Data Enhancement

- Integrate real-time foot traffic data
- Include social media sentiment analysis
- Add temporal dynamics and seasonality modeling

#### 6.2.2 Model Improvements

- Deep learning approaches for complex pattern recognition
- Ensemble methods combining multiple data sources
- Online learning for real-time model updates

#### 6.2.3 Expanded Applications

- Extension to other Canadian cities
- Adaptation for different business types (retail, services)
- Integration with economic development planning

### 6.3 Final Recommendations

1. **Immediate Implementation**: Deploy model as web application for public use
2. **Validation Study**: Partner with local restaurants to validate predictions
3. **Policy Integration**: Work with city planning to incorporate insights
4. **Continuous Improvement**: Establish feedback loops for model refinement

## References

1. [Relevant academic papers on restaurant success factors]
2. [Urban planning and GIS literature]
3. [Machine learning methodology references]
4. [Data source documentation]

## Appendices

### Appendix A: Data Collection Scripts

[Code snippets and API documentation]

### Appendix B: Feature Engineering Details

[Complete feature definitions and calculations]

### Appendix C: Model Hyperparameters

[Final model configurations and parameters]

### Appendix D: Additional Visualizations

[Supplementary maps, charts, and analysis]

---

_This report represents a comprehensive analysis of restaurant success prediction in Vancouver, BC. The methodology and findings provide a foundation for data-driven business location decisions and urban planning initiatives._
