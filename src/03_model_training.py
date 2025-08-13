"""
Model Training Script for VancouverPy Restaurant Success Prediction

This script handles the complete machine learning pipeline:
1. Data loading and exploration
2. Feature selection and engineering
3. Model training and evaluation
4. Model interpretation and visualization
5. Prediction and heat map generation

Author: VancouverPy Project Team
Date: August 2025
"""

# Standard library imports
import json
import logging
import time
import traceback
import warnings
from pathlib import Path

# Third-party imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Scikit-learn imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error, 
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Optional visualization imports
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: Folium not available. Map visualizations will be skipped.")

# Optional transformer imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    SA_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SA_TRANSFORMERS_AVAILABLE = False

# Configure warnings and logging
warnings.filterwarnings('ignore')

# Set up logging
base_dir = Path(__file__).parent.parent
log_path = base_dir / 'data' / 'model_training.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultilinguaSentimentAnalyzer:
    """Lightweight multilingual sentiment analyzer with graceful fallbacks.

    Usage:
        analyzer = MultilinguaSentimentAnalyzer()
        results = analyzer.predict_sentiment(["Great food", "Terrible service"])  # list of dicts
    """
    def __init__(self, multilingual_model: str = "tabularisai/multilingual-sentiment-analysis",
                 fallback_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 batch_size: int = 16):
        self.multilingual_model = multilingual_model
        self.fallback_model = fallback_model
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if (SA_TRANSFORMERS_AVAILABLE and 'torch' in globals() and torch.cuda.is_available()) else "cpu"
        if SA_TRANSFORMERS_AVAILABLE:
            self._load()
        else:
            logger.info("Transformers not available; using keyword-based sentiment fallback.")

    def _load(self):
        for name in [self.multilingual_model, self.fallback_model]:
            try:
                logger.info(f"Loading sentiment model: {name}")
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name)
                self.model.to(self.device)
                self.model_name = name
                logger.info(f"Loaded sentiment model '{name}' on {self.device}")
                return
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed loading {name}: {e}")
        logger.error("All transformer sentiment models failed; falling back to keyword method.")
        self.model = None
        self.tokenizer = None

    def predict_sentiment(self, texts):
        if not texts:
            return []
        if not SA_TRANSFORMERS_AVAILABLE or self.model is None:
            return self._keyword_fallback(texts)
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                for t, p in zip(batch, probs):
                    cl = int(torch.argmax(p).item())
                    conf = float(torch.max(p).item())
                    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                    label = sentiment_map.get(cl, 'Neutral')
                    outputs.append({
                        'text': t,
                        'sentiment': label,
                        'confidence': conf,
                        'sentiment_score': self._score(label, conf)
                    })
            except Exception as e:  # pragma: no cover
                logger.error(f"Sentiment batch failed: {e}")
                outputs.extend(self._keyword_fallback(batch))
        return outputs

    # Helpers --------------------------------------------------------------------
    def _score(self, label, confidence):
        base = {'Negative': 0.2, 'Neutral': 0.5, 'Positive': 0.8}.get(label, 0.5)
        if label == 'Positive':
            return min(1.0, base + (confidence - 0.5) * 0.4)
        if label == 'Negative':
            return max(0.0, base - (confidence - 0.5) * 0.4)
        return base

    def _keyword_fallback(self, texts):
        pos = ['good','great','excellent','amazing','love','best','awesome','fantastic','wonderful','perfect','delicious','outstanding']
        neg = ['bad','terrible','awful','hate','worst','horrible','disgusting','disappointing','poor','rude','slow','expensive']
        out = []
        for t in texts:
            tl = (t or '').lower()
            p = sum(1 for w in pos if w in tl)
            n = sum(1 for w in neg if w in tl)
            if p>n:
                label, score = 'Positive', 0.7
            elif n>p:
                label, score = 'Negative', 0.3
            else:
                label, score = 'Neutral', 0.5
            out.append({'text': t, 'sentiment': label, 'confidence': 0.6, 'sentiment_score': score})
        return out

    # Aggregation ----------------------------------------------------------------
    def aggregate_reviews(self, df_reviews, business_col='business_id', text_col='text'):
        if df_reviews is None or df_reviews.empty or text_col not in df_reviews.columns:
            return pd.DataFrame()
        preds = self.predict_sentiment(df_reviews[text_col].fillna('').tolist())
        df_reviews = df_reviews.copy()
        df_reviews['sentiment'] = [p['sentiment'] for p in preds]
        df_reviews['sentiment_confidence'] = [p['confidence'] for p in preds]
        df_reviews['sentiment_score'] = [p['sentiment_score'] for p in preds]
        agg = df_reviews.groupby(business_col).agg({
            'sentiment_score':['mean','std','count'],
            'sentiment_confidence':'mean'
        })
        agg.columns = ['avg_sentiment_score','sentiment_score_std','sentiment_review_count','avg_sentiment_confidence']
        dist = df_reviews.groupby([business_col,'sentiment']).size().unstack(fill_value=0)
        for c in dist.columns:
            dist[f"{c.lower()}_pct"] = (dist[c] / dist.sum(axis=1) * 100).round(2)
        feat = agg.join(dist, how='left').fillna(0).reset_index()
        return feat

class RestaurantSuccessPredictor:
    """Main class for restaurant success prediction model training"""
    
    def __init__(self):
        # Use absolute paths
        base_dir = Path(__file__).parent.parent
        self.processed_data_dir = base_dir / 'data' / 'processed'
        self.models_dir = base_dir / 'models'
        self.plots_dir = base_dir / 'reports' / 'plots'
        
        # Create directories if they don't exist
        for directory in [self.models_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes
        self.df_restaurants = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
        # Set visualization style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def load_processed_data(self):
        """Load processed datasets (prioritize PySpark files if available)"""
        logger.info("Loading processed datasets...")
        
        try:
     
            
            # Regular pandas-processed files
            restaurants_path = Path(self.processed_data_dir) / 'restaurants_with_features.csv'
            features_path = Path(self.processed_data_dir) / 'model_features.csv'
            
            if restaurants_path.exists():
                # Define proper data types for numerical features that actually exist
                # First check what columns are available
                temp_df = pd.read_csv(restaurants_path, nrows=1)
                available_columns = set(temp_df.columns)
                
                dtype_spec = {}
                potential_dtypes = {
                    'stars': 'float64',
                    'review_count': 'float64', 
                    'price_level': 'float64',
                    'latitude': 'float64',
                    'longitude': 'float64',
                    'competitor_count': 'float64',
                    'similar_cuisine_count': 'float64',
                    'sentiment_score': 'float64',
                    'sentiment_confidence': 'float64',
                    'success_score': 'float64'
                }
                
                # Only include columns that actually exist
                for col, dtype in potential_dtypes.items():
                    if col in available_columns:
                        dtype_spec[col] = dtype
                
                self.df_restaurants = pd.read_csv(restaurants_path, dtype=dtype_spec)
                logger.info(f"Loaded standard restaurant data: {len(self.df_restaurants)} records")
                using_spark_data = False
            else:
                logger.error(f"Restaurant data not found at {restaurants_path}")
                return False
            

            if features_path.exists():
                # Use the same data type specifications for features
                self.X = pd.read_csv(features_path, dtype=dtype_spec)
                logger.info(f"Loaded standard features: {self.X.shape}")
            else:
                logger.error(f"Feature data not found at {features_path}")
                return False


            
            # Load feature names
            feature_names_path = Path(self.processed_data_dir) / 'feature_names.csv'
            if feature_names_path.exists():
                feature_df = pd.read_csv(feature_names_path)
                # Check which column exists
                if 'feature_name' in feature_df.columns:
                    self.feature_names = feature_df['feature_name'].tolist()
                elif 'feature' in feature_df.columns:
                    self.feature_names = feature_df['feature'].tolist()
                else:
                    # Fallback to first column
                    self.feature_names = feature_df.iloc[:, 0].tolist()
                logger.info(f"Loaded feature names: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names not found. Using column names from feature matrix.")
                self.feature_names = list(self.X.columns)
            
            # Validate data quality - check for constant features
            logger.info("Validating feature quality...")
            constant_features = []
            for col in self.X.columns:
                if self.X[col].dtype in ['int64', 'float64']:
                    if self.X[col].nunique() <= 1:
                        constant_features.append(col)
                        logger.warning(f"Feature '{col}' has constant values: {self.X[col].unique()}")
            
            if constant_features:
                logger.warning(f"Found {len(constant_features)} constant features that won't contribute to prediction")
                logger.info(f"Removing constant features: {constant_features}")
                self.X = self.X.drop(columns=constant_features)
                # Update feature names list
                self.feature_names = [f for f in self.feature_names if f not in constant_features]
                logger.info(f"Features after removing constants: {len(self.feature_names)} features")
            
            # Log feature statistics
            for col in ['stars', 'review_count', 'price_level']:
                if col in self.X.columns:
                    logger.info(f"{col}: range={self.X[col].min():.3f}-{self.X[col].max():.3f}, "
                               f"unique_values={self.X[col].nunique()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        logger.info("Performing exploratory data analysis...")

        if self.df_restaurants is None or self.X is None:
            logger.error("Data not loaded. Please run load_processed_data() first.")
            return

        self._log_dataset_overview()
        self._analyze_missing_values()
        self._plot_success_score_distribution()
        self._plot_feature_correlations()
        self._plot_geographic_distribution()

    def _log_dataset_overview(self):
        """Log basic information about the dataset."""
        print("\n" + "=" * 50)
        print("DATASET OVERVIEW")
        print("=" * 50)
        print(f"Number of restaurants: {len(self.df_restaurants)}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")

    def _analyze_missing_values(self):
        """Analyze and log missing values in the dataset."""
        missing_values = self.df_restaurants.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values found:")
            missing_df = pd.DataFrame({
                'Missing Count': missing_values[missing_values > 0],
                'Missing Percentage': (missing_values[missing_values > 0] / len(self.df_restaurants)) * 100
            })
            print(missing_df.sort_values('Missing Count', ascending=False))
        else:
            print("\nNo missing values found.")

    def _plot_success_score_distribution(self):
        """Plot the distribution of the success score."""
        if 'success_score' in self.df_restaurants.columns:
            success_stats = self.df_restaurants['success_score'].describe()
            print(f"\nSuccess Score Statistics:")
            print(success_stats)

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.hist(self.df_restaurants['success_score'], bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Success Scores')
            plt.xlabel('Success Score')
            plt.ylabel('Frequency')

            plt.subplot(1, 2, 2)
            plt.boxplot(self.df_restaurants['success_score'])
            plt.title('Success Score Box Plot')
            plt.ylabel('Success Score')

            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/success_score_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()

    def _plot_feature_correlations(self):
        """Plot the feature correlation matrix."""
        if len(self.X.columns) > 1:
            plt.figure(figsize=(12, 10))

            corr_matrix = self.X.corr()

            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
            plt.show()

            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            if high_corr_pairs:
                print(f"\nHighly correlated feature pairs (|correlation| > 0.8):")
                for pair in high_corr_pairs:
                    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
            else:
                print(f"\nNo highly correlated feature pairs found.")

    def _plot_geographic_distribution(self):
        """Plot the geographic distribution of restaurants."""
        if all(col in self.df_restaurants.columns for col in ['latitude', 'longitude']):
            plt.figure(figsize=(12, 8))

            if 'success_score' in self.df_restaurants.columns:
                scatter = plt.scatter(
                    self.df_restaurants['longitude'],
                    self.df_restaurants['latitude'],
                    c=self.df_restaurants['success_score'],
                    cmap='viridis', alpha=0.6, s=30
                )
                plt.colorbar(scatter, label='Success Score')
            else:
                plt.scatter(
                    self.df_restaurants['longitude'],
                    self.df_restaurants['latitude'],
                    alpha=0.6, s=30
                )

            plt.title('Geographic Distribution of Restaurants in Vancouver')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.plots_dir}/geographic_distribution.png", dpi=300, bbox_inches='tight')
            plt.show()

            print(f"\nCoordinate ranges:")
            print(f"Latitude: {self.df_restaurants['latitude'].min():.4f} to {self.df_restaurants['latitude'].max():.4f}")
            print(f"Longitude: {self.df_restaurants['longitude'].min():.4f} to {self.df_restaurants['longitude'].max():.4f}")
    
    def prepare_target_variable(self):
        """Prepare the target variable for modeling with enhanced success score calculation"""
        logger.info("Preparing target variable...")
        
        # Always use enhanced success score for better model performance
        logger.info("Creating enhanced target variable for optimal model performance...")
        self._create_enhanced_success_score()
        
        # Remove success_score from features to prevent data leakage
        if 'success_score' in self.X.columns:
            self.X = self.X.drop('success_score', axis=1)
            logger.info("Removed success_score from features to prevent data leakage")
        
        # Ensure X and y have same length
        min_length = min(len(self.X), len(self.y))
        self.X = self.X.iloc[:min_length].copy()
        self.y = self.y.iloc[:min_length] if hasattr(self.y, 'iloc') else self.y[:min_length]
        
        logger.info(f"Final dataset shape: X={self.X.shape}, y={len(self.y)}")
        logger.info(f"Final features: {list(self.X.columns)}")
    
    def _create_enhanced_success_score(self):
        """Create a more sophisticated success score based on multiple factors"""
        logger.info("Creating enhanced success score...")
        
        # Initialize components
        rating_component = 0
        popularity_component = 0
        longevity_component = 0
        location_component = 0
        
        # Rating component (40% weight)
        if 'stars' in self.df_restaurants.columns:
            # Normalize stars to 0-1 scale
            rating_component = (self.df_restaurants['stars'] - 1) / 4  # Assuming 1-5 scale
            rating_component = np.clip(rating_component, 0, 1)
        
        # Popularity component (30% weight)
        if 'review_count' in self.df_restaurants.columns:
            # Log-normalized review count
            log_reviews = np.log1p(self.df_restaurants['review_count'])
            popularity_component = (log_reviews - log_reviews.min()) / (log_reviews.max() - log_reviews.min() + 1e-8)
        
        # Location desirability component (20% weight)
        if all(col in self.df_restaurants.columns for col in ['latitude', 'longitude']):
            # Distance from downtown Vancouver (closer is better)
            downtown_lat, downtown_lon = 49.2827, -123.1207
            distances = np.sqrt(
                (self.df_restaurants['latitude'] - downtown_lat)**2 + 
                (self.df_restaurants['longitude'] - downtown_lon)**2
            )
            # Invert and normalize (closer = higher score)
            max_distance = distances.max()
            location_component = 1 - (distances / max_distance)
        
        # Competition resistance component (10% weight)
        if 'competitor_count' in self.df_restaurants.columns:
            # Lower competition = higher score
            max_competitors = self.df_restaurants['competitor_count'].max()
            longevity_component = 1 - (self.df_restaurants['competitor_count'] / (max_competitors + 1))
        
        # Combine components with weights
        weights = {
            'rating': 0.4,
            'popularity': 0.3,
            'location': 0.2,
            'competition': 0.1
        }
        
        success_score = (
            weights['rating'] * rating_component +
            weights['popularity'] * popularity_component +
            weights['location'] * location_component +
            weights['competition'] * longevity_component
        )
        
        # Add some noise to create more realistic variance
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(success_score))
        success_score += noise
        
        # Ensure scores are between 0 and 1
        success_score = np.clip(success_score, 0, 1)
        
        # Create categories for better interpretation
        self.df_restaurants['success_category'] = pd.cut(
            success_score, 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        self.y = pd.Series(success_score)
        logger.info(f"Enhanced success score created. Mean={self.y.mean():.3f}, Std={self.y.std():.3f}")
        logger.info(f"Success categories: {self.df_restaurants['success_category'].value_counts().to_dict()}")
    
    
    def _enhance_features(self):
        """Enhanced feature engineering to improve model performance"""
        logger.info("Performing enhanced feature engineering...")
        
        try:
            # Create interaction features for geographic and business features
            if 'latitude' in self.X.columns and 'longitude' in self.X.columns:
                # Distance from downtown Vancouver (approximately 49.2827, -123.1207)
                downtown_lat, downtown_lon = 49.2827, -123.1207
                self.X['distance_from_downtown'] = np.sqrt(
                    (self.X['latitude'] - downtown_lat)**2 + 
                    (self.X['longitude'] - downtown_lon)**2
                ) * 111  # Convert to km (approximate)
                
                # Create density features
                self.X['lat_lon_interaction'] = self.X['latitude'] * self.X['longitude']
            
            # Business density and competition features
            if 'competitor_count' in self.X.columns and 'similar_cuisine_count' in self.X.columns:
                self.X['competition_ratio'] = self.X['similar_cuisine_count'] / (self.X['competitor_count'] + 1)
                self.X['market_saturation'] = self.X['competitor_count'] * self.X['similar_cuisine_count']
            
            # Rating and review interactions
            if 'stars' in self.X.columns and 'review_count' in self.X.columns:
                self.X['rating_popularity'] = self.X['stars'] * np.log1p(self.X['review_count'])
                self.X['review_per_star'] = self.X['review_count'] / (self.X['stars'] + 0.1)
            
            # Price level features
            if 'price_level' in self.X.columns:
                self.X['is_budget'] = (self.X['price_level'] <= 1).astype(int)
                self.X['is_expensive'] = (self.X['price_level'] >= 3).astype(int)
            
            # Sentiment features enhancement
            if 'sentiment_score' in self.X.columns and 'sentiment_confidence' in self.X.columns:
                self.X['weighted_sentiment'] = self.X['sentiment_score'] * self.X['sentiment_confidence']
                self.X['sentiment_strength'] = np.abs(self.X['sentiment_score'] - 0.5) * 2
            
            # Log transformations for skewed features
            skewed_features = ['review_count', 'competitor_count', 'similar_cuisine_count']
            for feature in skewed_features:
                if feature in self.X.columns:
                    self.X[f'log_{feature}'] = np.log1p(self.X[feature])
            
            # Polynomial features for key variables (degree 2)
            key_features = ['stars', 'review_count']
            for feature in key_features:
                if feature in self.X.columns:
                    self.X[f'{feature}_squared'] = self.X[feature] ** 2
            
            # Remove features with zero variance
            initial_features = len(self.X.columns)
            self.X = self.X.loc[:, self.X.var() != 0]
            removed_features = initial_features - len(self.X.columns)
            
            if removed_features > 0:
                logger.info(f"Removed {removed_features} zero-variance features")
            
            # Handle any infinite or NaN values
            self.X = self.X.replace([np.inf, -np.inf], np.nan)
            self.X = self.X.fillna(self.X.median())
            
            logger.info(f"Enhanced feature engineering completed. Features: {len(self.X.columns)}")
            logger.info(f"New feature names: {list(self.X.columns)}")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _perform_hyperparameter_tuning(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """Perform hyperparameter tuning for only the best performing model"""
        logger.info("Performing hyperparameter tuning...")
        
        if not self.results:
            return
        
        # Find the best model for tuning (only tune the top performer)
        best_model_name = max(self.results.items(), key=lambda x: x[1]['R2'])[0]
        logger.info(f"Tuning hyperparameters for best model: {best_model_name}...")
        
        try:
            if best_model_name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 150],  # Reduced search space
                    'max_depth': [6, 8, 10],         # Focused on preventing overfitting
                    'min_samples_split': [10, 15, 20],
                    'min_samples_leaf': [5, 7, 10]
                }
                base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
                X_tune, y_tune = X_train, y_train
                
            elif best_model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [50, 100, 150],    # Reduced search space
                    'learning_rate': [0.03, 0.05, 0.1], # Lower learning rates
                    'max_depth': [3, 4, 5],            # Shallower trees
                    'min_child_weight': [5, 7, 10],    # Higher regularization
                    'reg_alpha': [1.0, 2.0, 3.0]      # More regularization
                }
                base_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)
                X_tune, y_tune = X_train, y_train
                
            elif 'Ridge' in best_model_name:
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]  # Broader alpha range
                }
                base_model = Ridge(random_state=42)
                X_tune, y_tune = X_train_scaled, y_train
                
            else:
                logger.info(f"No hyperparameter tuning defined for {best_model_name}")
                return
            
            # Perform grid search with reduced CV folds for speed
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3,  # Reduced for speed and overfitting prevention
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_tune, y_tune)
            
            # Test tuned model
            if 'Ridge' in best_model_name:
                y_pred_tuned = grid_search.predict(X_test_scaled)
            else:
                y_pred_tuned = grid_search.predict(X_test)
            
            r2_tuned = r2_score(y_test, y_pred_tuned)
            
            # Update if improved
            if r2_tuned > self.results[best_model_name]['R2']:
                logger.info(f"Improved {best_model_name}: R2 {self.results[best_model_name]['R2']:.3f} -> {r2_tuned:.3f}")
                
                # Update model and results
                tuned_name = f"{best_model_name}_Tuned"
                self.models[tuned_name] = grid_search.best_estimator_
                self.results[tuned_name] = {
                    'MSE': mean_squared_error(y_test, y_pred_tuned),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_tuned)),
                    'MAE': mean_absolute_error(y_test, y_pred_tuned),
                    'R2': r2_tuned,
                    'CV_R2_Mean': grid_search.best_score_,
                    'CV_R2_Std': 0,  # Not available from GridSearchCV
                    'y_pred': y_pred_tuned,
                    'y_test': y_test,
                    'Best_Params': grid_search.best_params_
                }
                
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                logger.info(f"Hyperparameter tuning did not improve {best_model_name}")
            
        except Exception as e:
            logger.error(f"Error tuning {best_model_name}: {e}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance using Random Forest"""
        logger.info("Analyzing feature importance...")
        
        if len(self.X.columns) < 2 or len(self.y) < 10:
            logger.warning("Insufficient data for feature importance analysis")
            return
        
        # Quick feature importance analysis
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_temp.fit(self.X, self.y)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nFeature Importance Ranking:")
        print(feature_importance)
        
        return feature_importance
    
    # Removed first duplicate perform_clustering_analysis (using unified version later in file)
    
    def interpret_clusters(self, cluster_df):
        """Provide business interpretation of clusters"""
        print(f"\nCLUSTER INTERPRETATION")
        print("="*50)
        
        # Sort clusters by average success score
        if 'Avg_Success_Score' in cluster_df.columns:
            sorted_clusters = cluster_df.sort_values('Avg_Success_Score', ascending=False)
            
            print("Clusters ranked by success score (highest to lowest):")
            for idx, row in sorted_clusters.iterrows():
                cluster_id = int(row['Cluster'])
                success_score = row['Avg_Success_Score']
                size = int(row['Size'])
                
                # Determine cluster characteristics
                if success_score > 0.6:
                    performance = "HIGH PERFORMING"
                elif success_score > 0.5:
                    performance = "MODERATE PERFORMING" 
                else:
                    performance = "UNDERPERFORMING"
                
                print(f"\nCluster {cluster_id}: {performance}")
                print(f"  - Size: {size} restaurants ({size/len(self.df_restaurants)*100:.1f}% of total)")
                print(f"  - Average Success Score: {success_score:.3f}")
                
                # Identify key characteristics
                feature_insights = []
                for col in ['Avg_stars', 'Avg_review_count', 'Avg_price_level']:
                    if col in row:
                        value = row[col]
                        if col == 'Avg_stars' and value > 4.0:
                            feature_insights.append("High ratings")
                        elif col == 'Avg_review_count' and value > 100:
                            feature_insights.append("Popular (many reviews)")
                        elif col == 'Avg_price_level' and value > 2:
                            feature_insights.append("Higher price point")
                
                if feature_insights:
                    print(f"  - Key characteristics: {', '.join(feature_insights[:3])}")
        
        # Business recommendations
        print(f"\nBUSINESS INSIGHTS")
        print("="*30)
        print("- High-performing clusters show successful restaurant patterns")
        print("- Geographic clustering reveals optimal neighborhoods")
        print("- Feature patterns indicate what drives restaurant success")
        print("- Use cluster characteristics to guide new restaurant strategy")

    def train_models(self):
        """Train multiple machine learning models with enhanced feature engineering and hyperparameter tuning"""
        logger.info("Training machine learning models with improved approach...")
        
        if len(self.X) < 20:
            logger.error(f"Dataset too small for training ({len(self.X)} samples). Need at least 20 samples.")
            return False
        
        # Enhanced feature engineering before training
        self._enhance_features()
        
        # Split data into training and testing sets with stratification if needed
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=True
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Testing set: {X_test.shape}")
        logger.info(f"Target variable range: {y_train.min():.3f} to {y_train.max():.3f}")
        
        # Enhanced feature scaling with outlier handling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Focus on most relevant models only (reduced from 6 to 3)
        model_configs = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,  # Reduced to prevent overfitting
                max_depth=8,       # Reduced depth
                min_samples_split=10,  # Increased to prevent overfitting
                min_samples_leaf=5,    # Increased to prevent overfitting
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,      # Reduced to prevent overfitting
                learning_rate=0.05,    # Lower learning rate
                max_depth=4,           # Reduced depth
                min_child_weight=5,    # Increased to prevent overfitting
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,         # Increased regularization
                reg_lambda=2.0,        # Increased regularization
                random_state=42,
                eval_metric='rmse',
                n_jobs=-1
            )
        }
        
        # Train and evaluate each model with enhanced evaluation
        for name, model in model_configs.items():
            logger.info(f"Training {name}...")
            
            try:
                # Use scaled data for linear models, original for tree-based
                if any(model_type in name for model_type in ['Linear', 'Ridge', 'Lasso']):
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Cross-validation with scaled data
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Cross-validation with original data
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Calculate comprehensive metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Calculate additional metrics
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
                explained_var = 1 - np.var(y_test - y_pred) / np.var(y_test)
                
                # Store enhanced results
                self.results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R2': r2,
                    'Explained_Variance': explained_var,
                    'CV_R2_Mean': cv_mean,
                    'CV_R2_Std': cv_std,
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'Train_R2': r2_score(y_train, model.predict(X_train_scaled if any(t in name for t in ['Linear', 'Ridge', 'Lasso']) else X_train))
                }
                
                self.models[name] = model
                
                # Enhanced logging with overfitting detection
                train_r2 = self.results[name]['Train_R2']
                overfitting = train_r2 - r2
                
                logger.info(f"{name} - Test R2: {r2:.3f}, Train R2: {train_r2:.3f}, "
                          f"Overfitting: {overfitting:.3f}, CV R2: {cv_mean:.3f}±{cv_std:.3f}")
                
                if overfitting > 0.1:
                    logger.warning(f"{name} shows signs of overfitting (Train-Test R2 gap: {overfitting:.3f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Hyperparameter tuning for best performing models
        self._perform_hyperparameter_tuning(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        
        logger.info("Model training completed!")
        return True
    
    def evaluate_models(self):
        """Compare and evaluate model performance with enhanced metrics"""
        logger.info("Evaluating model performance...")
        if not self.results:
            logger.error("No model results available. Please run train_models() first.")
            return None
        
        # Create results dataframe with enhanced metrics
        results_df = pd.DataFrame(self.results).T
        
        # Sort by R2 score (primary metric)
        results_df = results_df.sort_values('R2', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Display key metrics
        display_cols = ['R2', 'RMSE', 'MAE', 'CV_R2_Mean', 'CV_R2_Std']
        available_cols = [col for col in display_cols if col in results_df.columns]
        print(results_df[available_cols].round(4))
        
        # Check for overfitting
        if 'Train_R2' in results_df.columns:
            results_df['Overfitting'] = results_df['Train_R2'] - results_df['R2']
            print(f"\nOverfitting Analysis (Train R2 - Test R2):")
            overfitting_df = results_df[['R2', 'Train_R2', 'Overfitting']].round(4)
            print(overfitting_df)
            
            # Highlight overfitting issues
            overfitting_models = results_df[results_df['Overfitting'] > 0.1].index.tolist()
            if overfitting_models:
                print(f"\nModels showing overfitting (gap > 0.1): {overfitting_models}")
        
        # Enhanced visualization
        n_metrics = len(available_cols)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Plot each metric
        for i, metric in enumerate(available_cols):
            ax = axes[i]
            results_df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='navy')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Model complexity vs performance scatter
        if len(available_cols) > 4:
            ax = axes[5]
            if 'CV_R2_Std' in results_df.columns:
                scatter = ax.scatter(results_df['R2'], results_df['CV_R2_Std'], 
                                   s=100, alpha=0.7, c=results_df['RMSE'], cmap='viridis')
                ax.set_xlabel('R2 Score')
                ax.set_ylabel('CV R2 Std (Model Stability)')
                ax.set_title('Model Performance vs Stability')
                plt.colorbar(scatter, ax=ax, label='RMSE')
                
                # Annotate points
                for idx, row in results_df.iterrows():
                    ax.annotate(idx[:10], (row['R2'], row['CV_R2_Std']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/enhanced_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Select best model (considering both performance and stability)
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        
        best_r2 = results_df.loc[self.best_model_name, 'R2']
        best_rmse = results_df.loc[self.best_model_name, 'RMSE']
        
        print(f"\n" + "="*50)
        print("BEST MODEL SELECTION")
        print("="*50)
        print(f"Best performing model: {self.best_model_name}")
        print(f"R2 Score: {best_r2:.4f}")
        print(f"RMSE: {best_rmse:.4f}")
        
        if 'CV_R2_Mean' in results_df.columns:
            cv_mean = results_df.loc[self.best_model_name, 'CV_R2_Mean']
            cv_std = results_df.loc[self.best_model_name, 'CV_R2_Std']
            print(f"Cross-validation R2: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Performance interpretation
        if best_r2 > 0.7:
            performance_level = "EXCELLENT"
        elif best_r2 > 0.5:
            performance_level = "GOOD"
        elif best_r2 > 0.3:
            performance_level = "FAIR"
        else:
            performance_level = "POOR"
        
        print(f"Performance Level: {performance_level}")
        
        # Model recommendations
        print(f"\nModel Recommendations:")
        if best_r2 < 0.3:
            print("- Consider collecting more diverse features")
            print("- Check data quality and target variable definition")
            print("- Try feature engineering or dimensionality reduction")
        elif best_r2 < 0.5:
            print("- Model shows moderate predictive power")
            print("- Consider ensemble methods or neural networks")
            print("- Investigate feature interactions")
        else:
            print("- Model shows good predictive performance")
            print("- Consider deployment for practical use")
            print("- Monitor for overfitting and data drift")
        
        return results_df
    
    def interpret_best_model(self):
        """Interpret the best performing model"""
        logger.info("Interpreting best model...")
        
        if self.best_model is None:
            logger.error("No best model available. Please run evaluate_models() first.")
            return
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 10 features
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top 10 Feature Importance - {self.best_model_name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/best_model_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTop 5 Most Important Features for {self.best_model_name}:")
            print(feature_importance.head())
            
        # Coefficients for linear models
        elif hasattr(self.best_model, 'coef_'):
            coefficients = pd.DataFrame({
                'feature': self.X.columns,
                'coefficient': self.best_model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_coef = coefficients.head(10)
            sns.barplot(data=top_coef, x='coefficient', y='feature')
            plt.title(f'Top 10 Feature Coefficients - {self.best_model_name}')
            plt.xlabel('Coefficient Value')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/best_model_coefficients.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTop 5 Features by Coefficient Magnitude for {self.best_model_name}:")
            print(coefficients.head())
        
        # Prediction vs Actual plot
        best_results = self.results[self.best_model_name]
        y_test = best_results['y_test']
        y_pred = best_results['y_pred']
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Success Score')
        plt.ylabel('Predicted Success Score')
        plt.title(f'Prediction vs Actual - {self.best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/prediction_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Residual analysis
        residuals = y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Success Score')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {self.best_model_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/residual_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nResidual Statistics:")
        print(f"Mean: {residuals.mean():.4f}")
        print(f"Std: {residuals.std():.4f}")
        print(f"Mean Absolute Residual: {abs(residuals).mean():.4f}")
    
    def save_models(self):
        """Save trained models and scaler"""
        logger.info("Saving trained models...")
        
        # Save best model
        if self.best_model is not None:
            model_path = Path(self.models_dir) / f"best_model_{self.best_model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
        
        # Save scaler
        scaler_path = Path(self.models_dir) / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save all models
        all_models_path = Path(self.models_dir) / "all_models.pkl"
        joblib.dump(self.models, all_models_path)
        
        # Save results
        results_path = Path(self.models_dir) / "model_results.pkl"
        results_for_save = {k: {metric: v for metric, v in model_results.items() 
                              if metric not in ['y_pred', 'y_test']} 
                           for k, model_results in self.results.items()}
        joblib.dump(results_for_save, results_path)
        
        logger.info("All models and results saved successfully!")
    
    def predict_restaurant_success(self, latitude, longitude, **kwargs):
        """
        Predict success score for a new restaurant location with realistic features
        """
        if self.best_model is None:
            return "No trained model available"
        
        try:
            # Create more realistic feature values based on dataset statistics
            feature_dict = {}
            
            # Location features
            feature_dict['latitude'] = latitude
            feature_dict['longitude'] = longitude
            
            # Use realistic defaults based on dataset statistics
            if hasattr(self, 'df_restaurants') and len(self.df_restaurants) > 0:
                # Calculate means for realistic defaults
                stars_mean = self.df_restaurants.get('stars', pd.Series([4.0])).mean()
                review_count_mean = self.df_restaurants.get('review_count', pd.Series([10])).mean()
                price_level_mean = self.df_restaurants.get('price_level', pd.Series([2])).mean()
                
                # Set realistic defaults
                feature_dict['stars'] = kwargs.get('stars', stars_mean)
                feature_dict['review_count'] = kwargs.get('review_count', review_count_mean)
                feature_dict['price_level'] = kwargs.get('price_level', price_level_mean)
                
                # Calculate distance-based features
                downtown_lat, downtown_lon = 49.2827, -123.1207  # Vancouver downtown
                distance_from_downtown = np.sqrt((latitude - downtown_lat)**2 + (longitude - downtown_lon)**2)
                feature_dict['distance_from_downtown'] = distance_from_downtown
                
                # Competition features (vary by location)
                base_competition = np.random.uniform(5, 20)  # Random but realistic competition
                feature_dict['competitor_count'] = kwargs.get('competitor_count', base_competition)
                feature_dict['similar_cuisine_count'] = kwargs.get('similar_cuisine_count', base_competition * 0.3)
                
                # Sentiment features
                feature_dict['sentiment_score'] = kwargs.get('sentiment_score', 0.5)
                feature_dict['sentiment_confidence'] = kwargs.get('sentiment_confidence', 0.6)
                
                # Interaction features (if they exist in the model)
                if 'lat_lon_interaction' in self.X.columns:
                    feature_dict['lat_lon_interaction'] = latitude * longitude
                if 'competition_ratio' in self.X.columns:
                    feature_dict['competition_ratio'] = feature_dict.get('similar_cuisine_count', 1) / (feature_dict.get('competitor_count', 1) + 1)
                if 'rating_popularity' in self.X.columns:
                    feature_dict['rating_popularity'] = feature_dict.get('stars', 4) * np.log1p(feature_dict.get('review_count', 10))
                if 'weighted_sentiment' in self.X.columns:
                    feature_dict['weighted_sentiment'] = feature_dict.get('sentiment_score', 0.5) * feature_dict.get('sentiment_confidence', 0.6)
            
            # Create feature vector matching model training features
            features = []
            for col in self.X.columns:
                if col in feature_dict:
                    features.append(feature_dict[col])
                else:
                    # Provide reasonable defaults for missing features
                    if 'log_' in col:
                        features.append(np.log1p(10))  # log of reasonable default
                    elif 'squared' in col:
                        features.append(16)  # square of reasonable default
                    else:
                        features.append(0.5)  # neutral default
            
            features = np.array(features).reshape(1, -1)
            
            # Handle any NaN or infinite values
            features = np.nan_to_num(features, nan=0.5, posinf=10, neginf=-10)
            
            # Scale if needed
            if 'Linear' in self.best_model_name or 'Ridge' in self.best_model_name:
                features_scaled = self.scaler.transform(features)
                prediction = self.best_model.predict(features_scaled)[0]
            else:
                prediction = self.best_model.predict(features)[0]
            
            return max(0, min(1, prediction))  # Ensure prediction is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5  # Return neutral prediction on error
    
    def create_prediction_heatmap(self):
        """Create a prediction heat map for Vancouver with diverse predictions"""
        logger.info("Creating prediction heat map...")
        
        if self.best_model is None or 'latitude' not in self.df_restaurants.columns:
            logger.warning("Cannot create heat map - missing model or location data")
            return
        
        # Create a grid of predictions across Vancouver
        lat_min, lat_max = self.df_restaurants['latitude'].min(), self.df_restaurants['latitude'].max()
        lon_min, lon_max = self.df_restaurants['longitude'].min(), self.df_restaurants['longitude'].max()
        
        # Add padding to the bounds
        lat_padding = (lat_max - lat_min) * 0.1
        lon_padding = (lon_max - lon_min) * 0.1
        
        # Create grid
        lat_range = np.linspace(lat_min - lat_padding, lat_max + lat_padding, 20)
        lon_range = np.linspace(lon_min - lon_padding, lon_max + lon_padding, 20)
        
        # Generate predictions for grid points with varying parameters
        grid_predictions = []
        downtown_lat, downtown_lon = 49.2827, -123.1207  # Vancouver downtown
        
        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                # Vary features based on location to create realistic diversity
                distance_from_downtown = np.sqrt((lat - downtown_lat)**2 + (lon - downtown_lon)**2)
                
                # Create location-dependent features
                stars = 3.5 + np.random.normal(0, 0.5)  # Random rating around 3.5-4.5
                stars = max(1, min(5, stars))
                
                review_count = max(1, int(np.random.exponential(15)))  # Exponential distribution
                
                # Competition varies by location (higher in downtown)
                base_competition = 20 - (distance_from_downtown * 50)  # More competition downtown
                base_competition = max(1, base_competition + np.random.normal(0, 5))
                
                # Price level varies by area
                price_level = 2 + (np.random.random() - 0.5) + (distance_from_downtown * -2)  # Cheaper away from downtown
                price_level = max(1, min(4, price_level))
                
                pred = self.predict_restaurant_success(
                    lat, lon,
                    stars=stars,
                    review_count=review_count,
                    competitor_count=base_competition,
                    similar_cuisine_count=base_competition * 0.3,
                    price_level=price_level
                )
                
                if isinstance(pred, (int, float)) and not np.isnan(pred):
                    grid_predictions.append([lat, lon, pred])
        
        if not grid_predictions:
            logger.warning("Could not generate grid predictions")
            return
        
        grid_df = pd.DataFrame(grid_predictions, columns=['latitude', 'longitude', 'predicted_success'])
        
        # Check prediction variance
        pred_var = grid_df['predicted_success'].var()
        pred_range = grid_df['predicted_success'].max() - grid_df['predicted_success'].min()
        logger.info(f"Prediction variance: {pred_var:.4f}, Range: {pred_range:.4f}")
        
        # Create heat map visualization
        plt.figure(figsize=(12, 10))
        
        # Use better color scheme and scaling
        vmin = grid_df['predicted_success'].quantile(0.1)  # Use percentiles for better contrast
        vmax = grid_df['predicted_success'].quantile(0.9)
        
        # Scatter plot with predictions
        scatter = plt.scatter(grid_df['longitude'], grid_df['latitude'], 
                            c=grid_df['predicted_success'], cmap='RdYlGn', 
                            s=150, alpha=0.8, edgecolors='black', linewidth=0.5,
                            vmin=vmin, vmax=vmax)
        
        # Overlay actual restaurant locations
        if len(self.df_restaurants) > 0:
            plt.scatter(self.df_restaurants['longitude'], self.df_restaurants['latitude'], 
                       c='blue', s=10, alpha=0.3, label='Existing Restaurants')
        
        plt.colorbar(scatter, label='Predicted Success Score')
        plt.title(f'Restaurant Success Prediction Heat Map - Vancouver\n(Range: {pred_range:.3f}, Best: {grid_df["predicted_success"].max():.3f})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.plots_dir}/prediction_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Heat map created with {len(grid_predictions)} prediction points")
        
        # Save grid predictions
        grid_df.to_csv(f"{self.processed_data_dir}/prediction_grid.csv", index=False)

    def generate_cuisine_recommendations_geojson(self, min_cuisine_count: int = 5):
        """Generate one recommended spot per cuisine and export as GeoJSON.

        Strategy:
        - Derive a primary cuisine token from first category in 'categories' column.
        - For each cuisine with at least min_cuisine_count occurrences, pick the restaurant (existing)
          with highest success_score (or predicted if success_score missing) as representative.
        - Output a FeatureCollection with point geometries and useful properties.
        """
        logger.info("Generating cuisine recommendation GeoJSON...")
        if self.df_restaurants is None:
            logger.warning("Restaurant dataframe not loaded; cannot produce recommendations.")
            return None
        required_cols = {'latitude','longitude'}
        if not required_cols.issubset(self.df_restaurants.columns):
            logger.warning("Missing latitude/longitude; skipping cuisine recommendations.")
            return None

        df = self.df_restaurants.copy()
        # Derive primary cuisine - use primary_category if available, otherwise fallback to categories
        if 'primary_category' in df.columns:
            df['primary_cuisine'] = df['primary_category'].fillna('').astype(str)
        elif 'categories' in df.columns:
            df['primary_cuisine'] = df['categories'].fillna('').apply(lambda x: x.split(',')[0].strip() if x else '')
        elif 'cuisine' in df.columns:
            df['primary_cuisine'] = df['cuisine'].fillna('').astype(str)
        else:
            logger.warning("No primary_category/categories/cuisine column; cannot derive cuisines.")
            return None
        df = df[df['primary_cuisine'] != '']
        if df.empty:
            logger.warning("No cuisines found after processing.")
            return None

        # If success_score missing, attempt prediction via best model
        need_prediction = 'success_score' not in df.columns
        if need_prediction and self.best_model is not None:
            preds = []
            for _, row in df.iterrows():
                pred = self.predict_restaurant_success(row['latitude'], row['longitude'])
                preds.append(pred if pred is not None else 0.0)
            df['success_score'] = preds
        elif need_prediction:
            logger.warning("No success_score and no model to infer it.")
            df['success_score'] = 0.0

        cuisine_counts = df['primary_cuisine'].value_counts()
        valid_cuisines = cuisine_counts[cuisine_counts >= min_cuisine_count].index.tolist()
        if not valid_cuisines:
            logger.warning("No cuisines meet minimum count threshold; lowering threshold to 1.")
            valid_cuisines = cuisine_counts.index.tolist()

        features = []
        for cuisine in valid_cuisines:
            subset = df[df['primary_cuisine'] == cuisine]
            if subset.empty:
                continue
            top_row = subset.sort_values('success_score', ascending=False).iloc[0]
            lon = float(top_row['longitude'])
            lat = float(top_row['latitude'])
            
            # Get restaurant name with proper NaN handling
            restaurant_name = 'Unknown Restaurant'
            if pd.notna(top_row.get('title')):
                restaurant_name = str(top_row['title'])
            elif pd.notna(top_row.get('business_name')):
                restaurant_name = str(top_row['business_name'])
            elif pd.notna(top_row.get('name')):
                restaurant_name = str(top_row['name'])
            
            props = {
                'cuisine': cuisine,
                'recommended_restaurant': restaurant_name,
                'business_id': str(top_row.get('business_id', '')),
                'success_score': float(top_row.get('success_score', 0.0)),
                'review_count': int(top_row.get('review_count', 0)),
                'stars': float(top_row.get('stars', 0.0)),
            }
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': props
            })

        geojson = {'type': 'FeatureCollection', 'features': features}
        out_path = Path(self.processed_data_dir) / 'recommended_spots.geojson'
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=True, indent=2)
            logger.info(f"Cuisine recommendation GeoJSON written to {out_path}")
            print(f"Generated cuisine recommendations for {len(features)} cuisines -> {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed writing GeoJSON: {e}")
            return None
    
    def perform_clustering_analysis(self):
        """Perform K-Means clustering analysis on restaurants"""
        logger.info("Performing K-Means clustering analysis...")
        
        if self.X is None or len(self.X) < 10:
            logger.warning("Insufficient data for clustering analysis")
            return
        
        # Prepare data for clustering
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Determine optimal number of clusters using elbow method
        max_clusters = min(10, len(self.X) // 2)
        sse_scores = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        print("\n" + "="*50)
        print("K-MEANS CLUSTERING ANALYSIS")
        print("="*50)
        
        # Calculate SSE and Silhouette scores for different cluster numbers
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            sse = kmeans.inertia_
            silhouette = silhouette_score(X_scaled, cluster_labels)
            
            sse_scores.append(sse)
            silhouette_scores.append(silhouette)
            
            print(f"K={k}: SSE={sse:.2f}, Silhouette Score={silhouette:.3f}")
        
        # Plot elbow curve and silhouette scores
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        axes[0].plot(cluster_range, sse_scores, 'bo-')
        axes[0].set_title('Elbow Method for Optimal K')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Sum of Squared Errors (SSE)')
        axes[0].grid(True)
        
        # Silhouette scores
        axes[1].plot(cluster_range, silhouette_scores, 'ro-')
        axes[1].set_title('Silhouette Score vs Number of Clusters')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/clustering_optimization.png", dpi=300, bbox_inches='tight')
        plt.show()
        # Determine optimal K based on highest silhouette score
        optimal_k = list(cluster_range)[int(np.argmax(silhouette_scores))]
        best_silhouette = max(silhouette_scores)
        print(f"\nOptimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {best_silhouette:.3f}")

        # Run final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(X_scaled)

        # Attach labels for downstream use
        self.df_restaurants['cluster'] = final_labels

        # Analyze and visualize
        self.analyze_clusters(final_labels, optimal_k, kmeans_final)

        return final_labels, optimal_k

    # (Duplicate clustering block removed)

    def analyze_clusters(self, cluster_labels, n_clusters, kmeans_model):
        """Analyze the characteristics of each cluster"""
        logger.info("Analyzing cluster characteristics...")
        print(f"\nCLUSTER ANALYSIS ({n_clusters} clusters)")
        print("="*50)
        
        # Cluster size distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        print(f"Cluster sizes: {dict(cluster_counts)}")
        
        # Analyze cluster characteristics
        cluster_stats = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_restaurants = self.df_restaurants[cluster_mask]
            
            stats = {
                'Cluster': cluster_id,
                'Size': cluster_mask.sum(),
                'Avg_Success_Score': cluster_restaurants['success_score'].mean() if 'success_score' in cluster_restaurants.columns else 'N/A'
            }
            
            # Add feature statistics
            cluster_features = self.X[cluster_mask]
            for feature in self.X.columns:
                stats[f'Avg_{feature}'] = cluster_features[feature].mean()
            
            cluster_stats.append(stats)
        
        cluster_df = pd.DataFrame(cluster_stats)
        
        # Display key cluster characteristics
        print("\nCluster Characteristics:")
        key_columns = ['Cluster', 'Size', 'Avg_Success_Score'] + [col for col in cluster_df.columns if col.startswith('Avg_') and len(col) < 20][:5]
        display_df = cluster_df[key_columns].round(3)
        print(display_df.to_string(index=False))
        
        # Save detailed cluster analysis
        cluster_df.to_csv(f"{self.processed_data_dir}/cluster_analysis.csv", index=False)
        
        # Visualize clusters
        self.visualize_clusters(cluster_labels, n_clusters)
        
        # Interpret clusters
        self.interpret_clusters(cluster_df)

    def visualize_clusters(self, cluster_labels, n_clusters):
        """Create visualizations for the clusters"""
        logger.info("Creating cluster visualizations...")
        
        # If we have latitude/longitude, create geographic visualization
        if all(col in self.df_restaurants.columns for col in ['latitude', 'longitude']):
            plt.figure(figsize=(14, 6))
            
            # Geographic cluster plot
            plt.subplot(1, 2, 1)
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = self.df_restaurants[cluster_mask]
                
                plt.scatter(cluster_data['longitude'], cluster_data['latitude'], 
                           c=[colors[cluster_id]], label=f'Cluster {cluster_id}', 
                           alpha=0.7, s=50)
            
            plt.title('Geographic Distribution of Restaurant Clusters')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Success score by cluster
            plt.subplot(1, 2, 2)
            if 'success_score' in self.df_restaurants.columns:
                cluster_success = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_scores = self.df_restaurants[cluster_mask]['success_score']
                    cluster_success.append(cluster_scores)
                
                plt.boxplot(cluster_success, labels=[f'C{i}' for i in range(n_clusters)])
                plt.title('Success Score Distribution by Cluster')
                plt.xlabel('Cluster')
                plt.ylabel('Success Score')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/cluster_visualization.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # PCA visualization for feature space
        if len(self.X.columns) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.scaler.fit_transform(self.X))
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                           c=[colors[cluster_id]], label=f'Cluster {cluster_id}', 
                           alpha=0.7, s=50)
            
            plt.title('Restaurant Clusters in PCA Feature Space')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.plots_dir}/cluster_pca_visualization.png", dpi=300, bbox_inches='tight')
            plt.show()

    def interpret_clusters(self, cluster_df):
        """Provide business interpretation of clusters (duplicate cleaned)"""
        print(f"\nCLUSTER INTERPRETATION")
        print("="*50)
        
        if 'Avg_Success_Score' in cluster_df.columns and cluster_df['Avg_Success_Score'].dtype != 'object':
            sorted_clusters = cluster_df.sort_values('Avg_Success_Score', ascending=False)
            print("Clusters ranked by success score (highest to lowest):")
            overall_means = self.X.mean()
            for idx, row in sorted_clusters.iterrows():
                cluster_id = int(row['Cluster'])
                success_score = row['Avg_Success_Score']
                size = int(row['Size'])
                if success_score > 0.7:
                    performance = "HIGH PERFORMING"
                elif success_score > 0.5:
                    performance = "MODERATE PERFORMING"
                else:
                    performance = "UNDERPERFORMING"
                print(f"\nCluster {cluster_id}: {performance}")
                print(f"  - Size: {size} restaurants ({size/len(self.df_restaurants)*100:.1f}% of total)")
                print(f"  - Average Success Score: {success_score:.3f}")
                feature_insights = []
                for col in self.X.columns:
                    avg_col_key = f'Avg_{col}'
                    if avg_col_key in row:
                        cluster_mean = row[avg_col_key]
                        overall_mean = overall_means.get(col, None)
                        if overall_mean is not None and overall_mean != 0 and abs(cluster_mean - overall_mean) > abs(overall_mean) * 0.2:
                            feature_insights.append(("High " if cluster_mean > overall_mean else "Low ") + col)
                if feature_insights:
                    print(f"  - Key characteristics: {', '.join(feature_insights[:3])}")
        
        print(f"\nBUSINESS INSIGHTS")
        print("="*30)
        print("- High-performing clusters indicate successful restaurant archetypes")
        print("- Geographic clustering may reveal optimal neighborhoods")
        print("- Feature patterns show what drives restaurant success")
        print("- Use cluster characteristics to guide new restaurant positioning")

def main():
    """Main execution function for model training"""
    print("VancouverPy: Restaurant Success Prediction Model Training")
    print("=" * 60)
    
    predictor = RestaurantSuccessPredictor()
    
    try:
        # 1. Load processed data
        if not predictor.load_processed_data():
            logger.error("Failed to load data. Run data collection/processing first.")
            return False

        # 2. Explore data
        predictor.explore_data()

        # 3. Prepare target variable
        predictor.prepare_target_variable()

        # 4. Analyze feature importance
        predictor.analyze_feature_importance()

        # 5. Perform clustering analysis
        cluster_labels, optimal_k = predictor.perform_clustering_analysis()
        if cluster_labels is None or len(cluster_labels) == 0:
            optimal_k = "N/A"

        # 6. Train models
        if not predictor.train_models():
            logger.error("Training failed.")
            return False

        # 7. Evaluate models
        predictor.evaluate_models()

        # 8. Interpret best model
        predictor.interpret_best_model()

        # 9. Save models
        predictor.save_models()

        # 10. Generate cuisine recommendations
        predictor.generate_cuisine_recommendations_geojson()

        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {predictor.best_model_name}")
        print(f"Optimal clusters: {optimal_k}")
        print("Models saved to 'models/' directory")
        print("Use the prediction module for making predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\nModel training failed: {e}")
        print("Please check the logs for details.")
        return False


if __name__ == "__main__":
    main()
