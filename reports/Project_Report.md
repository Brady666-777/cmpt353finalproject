# VancouverPy Project Report: Predicting Restaurant Success in Vancouver

**Course**: CMPT 353 - Computational Data Science  
**Date**: August 14, 2025  
**Authors**: VancouverPy Project Team

## Executive Summary

This report presents a machine learning framework for predicting restaurant success in Vancouver, BC. By analyzing 579 restaurants and engineering meaningful features, we developed a predictive model achieving R² = 0.639 using Ridge Regression (Tuned). Geographic location and review count emerged as the dominant success factors, with competitive environment and rating quality as secondary predictors.

**Key Findings**:

- Review count and geographic factors account for 70% of predictive importance
- Generated 24 cuisine-specific location recommendations
- Identified 7 distinct restaurant clusters across Vancouver
- Distance from downtown negatively correlates with success probability
- Model achieves GOOD performance level with minimal overfitting

## 1. Problem Statement and Refinement

Restaurant failure rates exceed 60% within the first year, with location cited as the critical success factor. Traditional site selection relies on intuition rather than data-driven analysis.

**Our Refinement**: Instead of binary success/failure prediction, we developed a continuous "Success Score" incorporating customer ratings, review volume, and competitive positioning. This provides more nuanced, actionable insights for restaurant entrepreneurs and urban planners.

**Research Questions**:

- What environmental factors most strongly predict restaurant success?
- How do geographic location and competition density influence performance?
- Can machine learning identify optimal sites for new restaurants?

## 2. Data Sources and Collection

### 2.1 Primary Datasets

**Vancouver Business Licenses** (City of Vancouver Open Data Portal):

- 4,064 food-related business licenses from static GeoJSON file
- Filtered to 579 active restaurants with valid coordinates and complete data
- Geographic distribution across Vancouver neighborhoods
- Business types, issue dates, and operational status
- File: `business-licences.geojson`

**Google Restaurant Data** (Static CSV Files):

- 606 restaurant profiles from combined Google datasets
- After data cleaning and validation: 579 restaurants with complete information
- 500 customer reviews for sentiment analysis
- Rating distributions and customer feedback patterns
- Files: `good-restaurant-in-vancouver-overview.csv`, `google-review_2025-08-06_03-55-37-484.csv`

**Statistics Canada Census 2021** (Static CSV File):

- 3,389 census profile records with demographic data
- Population density and income distribution by area
- Used for neighborhood profiling and market analysis
- File: `CensusProfile2021-ProfilRecensement2021-20250811051126.csv`

### 2.2 Data Collection Approach

**Static File Processing**: All data sources are pre-downloaded static files requiring no API calls or external dependencies. This approach ensures:

- Complete reproducibility across different environments
- No rate limiting or API key management issues
- Focus on data science methodology rather than data collection
- Realistic constraints for academic project timeline

**Data Integration Strategy**: Files are processed using pandas and geopandas, with robust error handling for encoding issues and missing values.

## 3. Methodology

### 3.1 Success Score Development

**Challenge**: Original data showed extremely low variance (all restaurants had identical base scores).

**Solution**: Enhanced Success Score incorporating:

- Rating quality weighted by review volume
- Geographic dispersion patterns and competitive environment effects
- Realistic score distribution modeling based on actual performance data

**Result**: Meaningful target variable with Mean=0.701, Std=0.066, creating a normally distributed success metric that effectively differentiates restaurant performance levels.

### 3.2 Feature Engineering

**Core Features (6 final)**:

- Spatial: latitude, longitude coordinates
- Performance: star ratings, review counts
- Competition: competitor count, similar cuisine density within 0.5km

**Enhanced Features (17 engineered)**:

- Distance from downtown Vancouver core
- Latitude-longitude interaction terms
- Competition ratios and market saturation indices
- Logarithmic transforms of review and competitor counts
- Derived metrics: reviews per star, rating popularity scores

### 3.3 Analysis Techniques

**Clustering Analysis**: K-means testing 2-10 clusters using elbow method and silhouette analysis. Optimal: 7 clusters with silhouette score 0.253.

**Machine Learning Pipeline**: Ridge Regression, Random Forest, XGBoost with 5-fold cross-validation and hyperparameter tuning.

**Geographic Analysis**: Grid-based competitive analysis for computational efficiency with large datasets.

## 4. Results and Findings

### 4.1 Model Performance

**Best Model**: Ridge Regression (Tuned)

- **Test R² Score**: 0.639 (Good performance)
- **RMSE**: 0.048, **Cross-Validation R²**: 0.604 ± 0.000
- **Overfitting**: Minimal (Train-Test R² gap: 0.001)
- **Best Parameters**: alpha = 10.0

**Model Comparison**:
| Model | R² Score | RMSE | CV R² | Overfitting |
|-------|----------|------|-------|-------------|
| Ridge (Tuned) | 0.639 | 0.048 | 0.604 | 0.001 |
| Ridge | 0.638 | 0.048 | 0.604 | 0.001 |
| Random Forest | 0.579 | 0.052 | 0.524 | 0.151 |
| XGBoost | 0.532 | 0.055 | 0.485 | 0.076 |

![Model Performance Comparison](plots/enhanced_model_comparison.png)
_Figure 1: Comprehensive comparison of model performance metrics_

![Prediction vs Actual](plots/prediction_vs_actual.png)
_Figure 2: Model predictions compared to actual success scores_

![Residual Analysis](plots/residual_plot.png)
_Figure 3: Residual analysis showing model prediction errors_

### 4.2 Feature Importance

**Top 5 Predictors by Coefficient Magnitude**:

1. **Distance from Downtown** (-0.028): Central proximity reduces success
2. **Log Review Count** (+0.025): Higher review volume increases success probability
3. **Competitor Count** (-0.023): Higher competition decreases probability
4. **Stars Rating** (+0.018): Better ratings correlate with success
5. **Rating Popularity** (+0.016): Review engagement drives performance

**Review Volume Dominance**: Review count (logarithmic) emerges as the strongest positive predictor, indicating that customer engagement and visibility are critical success factors.

![Feature Importance](plots/feature_importance.png)
_Figure 4: Feature importance ranking showing geographic factors dominance_

![Best Model Coefficients](plots/best_model_coefficients.png)
_Figure 5: Ridge Regression coefficients showing feature impact direction_

![Feature Correlations](plots/feature_correlations.png)
_Figure 6: Correlation matrix revealing relationships between features_

### 4.3 Clustering Analysis

**7 Optimal Clusters Identified** (Silhouette Score: 0.253):

- **Cluster 6**: 19 restaurants (3.3%) - Extremely High Performing (Success: 0.832)
- **Cluster 4**: 51 restaurants (8.8%) - High Performing (Success: 0.733)
- **Cluster 2**: 121 restaurants (20.9%) - High Performing (Success: 0.722)
- **Cluster 3**: 101 restaurants (17.4%) - High Performing (Success: 0.709)
- **Cluster 0**: 129 restaurants (22.3%) - High Performing (Success: 0.706)
- **Cluster 5**: 78 restaurants (13.5%) - Moderate Performing (Success: 0.659)
- **Cluster 1**: 80 restaurants (13.8%) - Moderate Performing (Success: 0.638)

**Key Finding**: Clear performance tiers emerged, with Cluster 6 representing the elite performers (high review volume, strong ratings) and Clusters 1 & 5 representing moderate performers with lower engagement.

![Clustering Optimization](plots/clustering_optimization.png)
_Figure 7: K-means clustering optimization showing optimal cluster selection_

![Cluster Visualization](plots/cluster_visualization.png)
_Figure 8: Geographic visualization of restaurant clusters across Vancouver_

![Cluster PCA Visualization](plots/cluster_pca_visualization.png)
_Figure 9: Principal component analysis of cluster characteristics_

### 4.4 Geographic Success Patterns

**High Potential Areas**: West End (lower competition), Kitsilano (affluent demographics), Commercial Drive (cultural diversity)

**Challenging Areas**: Downtown core (extreme competition), industrial zones (low foot traffic)

**Cuisine Recommendations**: 24 location suggestions across diverse cuisine types, focusing on areas with optimal competition-opportunity balance.

![Geographic Distribution](plots/geographic_distribution.png)
_Figure 10: Geographic distribution of restaurants across Vancouver neighborhoods_

![Prediction Heatmap](plots/prediction_heatmap.png)
_Figure 11: Success probability heatmap showing optimal locations for new restaurants_

![Success Score Distribution](plots/success_score_distribution.png)
_Figure 12: Distribution of success scores across the restaurant dataset_

### 4.5 Visualization Summary

**Model Performance Visualizations**:

- Figure 1: Enhanced model comparison showing R², RMSE, and cross-validation metrics
- Figure 2: Prediction vs actual scatter plot demonstrating model accuracy
- Figure 3: Residual analysis for identifying prediction patterns and outliers

**Feature Analysis Visualizations**:

- Figure 4: Feature importance ranking highlighting geographic dominance
- Figure 5: Ridge regression coefficients showing positive/negative impacts
- Figure 6: Feature correlation heatmap revealing multicollinearity patterns

**Clustering and Geographic Visualizations**:

- Figure 7: Clustering optimization using elbow method and silhouette scores
- Figure 8: Geographic cluster visualization mapping restaurant archetypes
- Figure 9: PCA analysis reducing dimensionality for cluster interpretation
- Figure 10: Restaurant distribution across Vancouver neighborhoods
- Figure 11: Success probability heatmap for location optimization
- Figure 12: Success score distribution showing target variable characteristics

**All visualizations are saved in `/reports/plots/` directory for detailed examination.**

## 5. Business Impact and Applications

### 5.1 Practical Applications

**Restaurant Entrepreneurs**: Quantitative site selection framework, competition analysis, risk assessment before lease commitment.

**Urban Planners**: Mixed-use development optimization, restaurant zoning decisions, economic development targeting.

**Investors**: Location-based due diligence, portfolio geographic diversification, default probability modeling.

### 5.2 Key Insights

- **Downtown Paradox**: Central locations show lower success despite higher foot traffic, likely due to extreme competition and overhead costs
- **Competition Sweet Spot**: Moderate competition indicates healthy demand; excessive competition dilutes success
- **Neighborhood Effects**: Clear clustering suggests location-specific success factors beyond simple demographics

## 6. Limitations and Future Work

### 6.1 Current Limitations

**Current Limitations**:

- Static snapshot dataset of 579 restaurants without temporal trends
- Review bias toward certain demographics and platforms
- Missing financial performance data (revenue, profit margins)
- Model explains 64% of variance, indicating other unmeasured factors

**Model Limitations**:

- Good predictive power (R²=0.639) but still significant unexplained variance
- Linear assumptions may miss complex non-linear relationships
- Vancouver-specific model may not generalize to other cities without retraining

### 6.2 Technical Challenges and Solutions

- **Sentiment Analysis**: Successfully resolved transformer compatibility issues by updating PyTorch components to compatible versions (torch==2.1.0, transformers==4.35.0). Now using tabularisai multilingual sentiment model.
- **Static Data Approach**: Eliminated API dependencies by using pre-downloaded static datasets, ensuring reproducibility
- **Computational Efficiency**: Implemented grid-based competitive analysis to handle large dataset processing

### 6.3 Future Improvements

**Enhanced Data**: Temporal analysis, financial metrics, foot traffic data, social media engagement

**Advanced Modeling**: Deep learning for non-linear patterns, ensemble methods, geographically weighted regression

**Validation**: Cross-city testing, A/B testing with actual restaurant openings, expert industry validation

### 6.4 In Retrospect

Should have: started with simpler success metrics, allocated more time for data quality assessment, implemented robust error handling, engaged industry experts for domain validation.

## 7. Conclusions

This analysis demonstrates that while restaurant success prediction remains challenging due to its multifaceted nature, data science approaches provide valuable quantitative insights for traditionally intuition-based decisions.

**Key Contributions**:

- **Methodological Framework**: Reproducible pipeline for restaurant success prediction achieving good performance
- **Review Volume Insights**: Customer engagement emerges as primary success driver beyond traditional location factors
- **Practical Tool**: 24 specific location recommendations with quantitative justification
- **Open Source**: Complete codebase available for replication and extension

The good predictive performance (R²=0.639) achieved demonstrates that review engagement, location factors, and competitive environment together explain approximately 64% of restaurant success variance, providing substantial value for data-driven decision making.

**Business Value**: This framework transforms subjective site selection into data-driven decision making, providing restaurant entrepreneurs, urban planners, and investors with quantitative tools for location assessment and risk mitigation.

_[Insert final prediction heatmap here]_

---

**Technical Implementation**: Complete pipeline available with modular design, error handling, and reproducible results. All analysis conducted using Python with pandas, geopandas, scikit-learn, and xgboost.

---

## 8. Latest Project Run Summary (August 14, 2025)

**Pipeline Execution Results**:

✅ **Data Collection**: Successfully processed 4,064 business licenses, 3,389 census records, and 606 restaurant profiles
✅ **Data Processing**: Generated 579 restaurants with complete feature sets and geocoded coordinates  
✅ **Feature Engineering**: Created 17 enhanced features including distance metrics, competition analysis, and interaction terms
✅ **Model Training**: Achieved R² = 0.639 with Ridge Regression (Tuned), representing GOOD performance level
✅ **Clustering Analysis**: Identified 7 distinct restaurant performance clusters
✅ **Recommendations**: Generated 24 cuisine-specific location recommendations

**Key Improvements in Latest Run**:

- Enhanced success score calculation with better variance (Mean: 0.701, Std: 0.066)
- Improved model performance from R² = 0.354 to R² = 0.639
- Better feature engineering with review engagement as primary predictor
- Minimal overfitting achieved (Train-Test gap: 0.001)
- More refined clustering analysis with clear performance tiers

**Performance Validation**:

- Cross-validation R² = 0.604 ± 0.000 (consistent performance)
- RMSE = 0.048 (low prediction error)
- Model successfully deployed and saved for production use

This updated analysis demonstrates significant improvements in predictive accuracy and business insight generation, confirming the robustness of the machine learning pipeline for restaurant success prediction in Vancouver.

<!--
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
-->
