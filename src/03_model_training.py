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

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import json
import time

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# Visualization imports
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: Folium not available. Map visualizations will be skipped.")

import logging

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

# ---------------------------------------------------------------------------
# Integrated Sentiment Analysis (formerly in sentiment_analysis.py)
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore
    SA_TRANSFORMERS_AVAILABLE = True
except Exception:
    SA_TRANSFORMERS_AVAILABLE = False

class MultilinguaSentimentAnalyzer:
    """Lightweight multilingual sentiment analyzer with graceful fallbacks.

    Usage:
        analyzer = MultilinguaSentimentAnalyzer()
        results = analyzer.predict_sentiment(["Great food", "Terrible service"])  # list of dicts
    """
    def __init__(self, multilingual_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                 fallback_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
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

    # Public API -----------------------------------------------------------------
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

class SentimentFeatureInjector:
    """Utility to add sentiment features on-the-fly if they are missing.
    This is a fallback for when upstream Spark processing wasn't run.
    It will stream through the raw Yelp review JSON and sample reviews to avoid OOM.
    """
    def __init__(self, base_dir: Path, max_reviews_per_business: int = 80, global_review_cap: int = 20000):
        self.base_dir = base_dir
        self.raw_dir = base_dir / 'data' / 'raw'
        self.review_path = self.raw_dir / 'yelp_academic_dataset_review.json'
        self.max_reviews_per_business = max_reviews_per_business
        self.global_review_cap = global_review_cap
        self.analyzer = MultilinguaSentimentAnalyzer()

    def add_if_missing(self, df_restaurants: pd.DataFrame) -> pd.DataFrame:
        required_cols = {'avg_sentiment_score','sentiment_score_std','avg_sentiment_confidence','positive_pct','negative_pct','neutral_pct'}
        if required_cols.issubset(df_restaurants.columns):
            logger.info("Sentiment features already present; skipping inline generation.")
            return df_restaurants
        if not self.review_path.exists():
            logger.warning("Review file not found; cannot compute sentiment features inline.")
            return df_restaurants
        if 'business_id' not in df_restaurants.columns:
            logger.warning("business_id column missing; cannot map sentiment features.")
            return df_restaurants
        business_ids = set(df_restaurants['business_id'].dropna().tolist())
        logger.info(f"Generating sentiment features inline for {len(business_ids)} businesses (streaming reviews)...")
        collected = []
        per_biz_counts = {bid:0 for bid in business_ids}
        total = 0
        start = time.time()
        try:
            with open(self.review_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if total >= self.global_review_cap and self.global_review_cap>0:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    bid = obj.get('business_id')
                    if bid in business_ids and per_biz_counts[bid] < self.max_reviews_per_business:
                        collected.append({'business_id': bid, 'text': obj.get('text','')})
                        per_biz_counts[bid] += 1
                        total += 1
            logger.info(f"Collected {total} review texts for inline sentiment in {time.time()-start:.1f}s")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed streaming reviews: {e}")
            return df_restaurants
        if not collected:
            logger.warning("No reviews collected for inline sentiment.")
            return df_restaurants
        df_reviews = pd.DataFrame(collected)
        feat = self.analyzer.aggregate_reviews(df_reviews)
        if feat.empty:
            logger.warning("No sentiment features produced.")
            return df_restaurants
        df_merged = df_restaurants.merge(feat, on='business_id', how='left')
        logger.info("Inline sentiment feature generation complete.")
        return df_merged

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
            # Check for PySpark-processed files first (higher quality data)
            spark_restaurants_path = Path(self.processed_data_dir) / 'restaurants_with_features_spark.csv'
            spark_features_path = Path(self.processed_data_dir) / 'model_features_spark.csv'
            
            # Regular pandas-processed files
            restaurants_path = Path(self.processed_data_dir) / 'restaurants_with_features.csv'
            features_path = Path(self.processed_data_dir) / 'model_features.csv'
            
            # Load main dataset (prefer PySpark if available)
            if spark_restaurants_path.exists():
                self.df_restaurants = pd.read_csv(spark_restaurants_path)
                logger.info(f"Loaded PySpark restaurant data: {len(self.df_restaurants)} records")
                using_spark_data = True
            elif restaurants_path.exists():
                self.df_restaurants = pd.read_csv(restaurants_path)
                logger.info(f"Loaded standard restaurant data: {len(self.df_restaurants)} records")
                using_spark_data = False
            else:
                logger.error(f"Restaurant data not found at {restaurants_path} or {spark_restaurants_path}")
                return False
            
            # Load feature matrix (prefer PySpark if available)
            if using_spark_data and spark_features_path.exists():
                self.X = pd.read_csv(spark_features_path)
                logger.info(f"Loaded PySpark features: {self.X.shape}")
            elif features_path.exists():
                self.X = pd.read_csv(features_path)
                logger.info(f"Loaded standard features: {self.X.shape}")
            else:
                logger.error(f"Feature data not found at {features_path} or {spark_features_path}")
                return False

            # Inline sentiment feature injection if missing
            sentiment_cols = {'avg_sentiment_score','sentiment_score_std','avg_sentiment_confidence','positive_pct','negative_pct','neutral_pct'}
            missing_sentiment = not sentiment_cols.issubset(set(self.df_restaurants.columns))
            if missing_sentiment:
                logger.info("Sentiment feature columns missing in restaurant data; attempting inline generation...")
                injector = SentimentFeatureInjector(Path(__file__).parent.parent)
                self.df_restaurants = injector.add_if_missing(self.df_restaurants)
                # If new sentiment columns appeared, append them to feature matrix
                new_cols = [c for c in sentiment_cols if c in self.df_restaurants.columns and c not in self.X.columns]
                if new_cols:
                    logger.info(f"Appending newly generated sentiment features to model matrix: {new_cols}")
                    # Join sentiment features by business_id if present in X or just align order
                    if 'business_id' in self.df_restaurants.columns:
                        if 'business_id' in self.X.columns:
                            # Merge on business_id
                            self.X = self.X.merge(self.df_restaurants[['business_id'] + new_cols], on='business_id', how='left')
                        else:
                            # Attach by index alignment (assume same order if lengths match)
                            if len(self.X) == len(self.df_restaurants):
                                for c in new_cols:
                                    self.X[c] = self.df_restaurants[c].values
                            else:
                                logger.warning("Cannot align sentiment features without business_id; skipping add.")
                    else:
                        logger.warning("business_id not in restaurant dataframe after injection; sentiment features not merged into X.")
                else:
                    logger.info("No new sentiment features were generated inline.")
            
            # Load feature names
            feature_names_path = Path(self.processed_data_dir) / 'feature_names.csv'
            if feature_names_path.exists():
                feature_df = pd.read_csv(feature_names_path)
                self.feature_names = feature_df['feature'].tolist()
                logger.info(f"Loaded feature names: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names not found. Using column names from feature matrix.")
                self.feature_names = list(self.X.columns)
            
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
        
        # Basic information
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Number of restaurants: {len(self.df_restaurants)}")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        
        # Missing values analysis
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
        
        # Success score analysis
        if 'success_score' in self.df_restaurants.columns:
            success_stats = self.df_restaurants['success_score'].describe()
            print(f"\nSuccess Score Statistics:")
            print(success_stats)
            
            # Plot success score distribution
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
            
        # Feature correlations
        if len(self.X.columns) > 1:
            plt.figure(figsize=(12, 10))
            
            # Calculate correlation matrix
            corr_matrix = self.X.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
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
        
        # Geographic distribution
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
        """Prepare the target variable for modeling"""
        logger.info("Preparing target variable...")
        
        if 'success_score' in self.df_restaurants.columns:
            self.y = self.df_restaurants['success_score'].copy()
            logger.info(f"Target variable prepared. Shape: {self.y.shape}")
            logger.info(f"Target statistics: Mean={self.y.mean():.3f}, Std={self.y.std():.3f}")
            
            # Remove success_score from features to prevent data leakage
            if 'success_score' in self.X.columns:
                self.X = self.X.drop('success_score', axis=1)
                logger.info("Removed success_score from features to prevent data leakage")
                
        else:
            logger.warning("Success score not found. Creating a dummy target for demonstration.")
            # Create dummy target for testing
            np.random.seed(42)
            self.y = np.random.normal(0.5, 0.2, len(self.df_restaurants))
            self.y = np.clip(self.y, 0, 1)  # Ensure values are between 0 and 1
            self.y = pd.Series(self.y)
        
        # Ensure X and y have same length
        min_length = min(len(self.X), len(self.y))
        self.X = self.X.iloc[:min_length].copy()
        self.y = self.y.iloc[:min_length] if hasattr(self.y, 'iloc') else self.y[:min_length]
        
        logger.info(f"Final dataset shape: X={self.X.shape}, y={len(self.y)}")
        logger.info(f"Final features: {list(self.X.columns)}")
    
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
        """Train multiple machine learning models"""
        logger.info("Training machine learning models...")
        
        if len(self.X) < 20:
            logger.error(f"Dataset too small for training ({len(self.X)} samples). Need at least 20 samples.")
            return False
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Testing set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        model_configs = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
        }
        
        # Train and evaluate each model
        for name, model in model_configs.items():
            logger.info(f"Training {name}...")
            
            try:
                # Use scaled data for linear models, original for tree-based
                if 'Linear' in name or 'Ridge' in name:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    # Use ASCII-safe metric keys to avoid Windows console encoding issues
                    'R2': r2,
                    'CV_R2_Mean': cv_mean,
                    'CV_R2_Std': cv_std,
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                self.models[name] = model
                
                # Avoid unicode superscript/plus-minus for Windows GBK consoles
                logger.info(f"{name} - R2: {r2:.3f}, RMSE: {rmse:.3f}, CV R2: {cv_mean:.3f}+/-{cv_std:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        logger.info("Model training completed!")
        return True
    
    def evaluate_models(self):
        """Compare and evaluate model performance"""
        logger.info("Evaluating model performance...")
        if not self.results:
            logger.error("No model results available. Please run train_models() first.")
            return None
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('R2', ascending=False)
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(results_df.round(4))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['R2', 'RMSE', 'MAE', 'MSE']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            results_df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        print(f"\nBest performing model: {self.best_model_name}")
        print(f"R2 Score: {results_df.loc[self.best_model_name, 'R2']:.3f}")
        print(f"RMSE: {results_df.loc[self.best_model_name, 'RMSE']:.3f}")
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
    
    def predict_restaurant_success(self, latitude, longitude, price_level=2, **kwargs):
        """
        Predict success score for a new restaurant location
        """
        if self.best_model is None:
            return "No trained model available"
        
        try:
            # Create feature vector (simplified - would need actual feature engineering)
            # This is a placeholder implementation
            features = np.array([[latitude, longitude, price_level] + [0] * (len(self.X.columns) - 3)])
            features = features[:, :len(self.X.columns)]  # Ensure correct number of features
            
            # Fill with provided kwargs or defaults
            feature_dict = dict(zip(self.X.columns, features[0]))
            for key, value in kwargs.items():
                if key in feature_dict:
                    feature_dict[key] = value
            
            features = np.array([list(feature_dict.values())]).reshape(1, -1)
            
            # Scale if needed
            if 'Linear' in self.best_model_name or 'Ridge' in self.best_model_name:
                features_scaled = self.scaler.transform(features)
                prediction = self.best_model.predict(features_scaled)[0]
            else:
                prediction = self.best_model.predict(features)[0]
            
            return max(0, min(1, prediction))  # Ensure prediction is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def create_prediction_heatmap(self):
        """Create a prediction heat map for Vancouver"""
        logger.info("Creating prediction heat map...")
        
        if self.best_model is None or 'latitude' not in self.df_restaurants.columns:
            logger.warning("Cannot create heat map - missing model or location data")
            return
        
        # Create a grid of predictions across Vancouver
        lat_min, lat_max = self.df_restaurants['latitude'].min(), self.df_restaurants['latitude'].max()
        lon_min, lon_max = self.df_restaurants['longitude'].min(), self.df_restaurants['longitude'].max()
        
        # Create grid
        lat_range = np.linspace(lat_min, lat_max, 20)
        lon_range = np.linspace(lon_min, lon_max, 20)
        
        # Generate predictions for grid points
        grid_predictions = []
        for lat in lat_range:
            for lon in lon_range:
                pred = self.predict_restaurant_success(lat, lon)
                if isinstance(pred, (int, float)):
                    grid_predictions.append([lat, lon, pred])
        
        if not grid_predictions:
            logger.warning("Could not generate grid predictions")
            return
        
        grid_df = pd.DataFrame(grid_predictions, columns=['latitude', 'longitude', 'predicted_success'])
        
        # Create heat map visualization
        plt.figure(figsize=(12, 10))
        
        # Scatter plot with predictions
        scatter = plt.scatter(grid_df['longitude'], grid_df['latitude'], 
                            c=grid_df['predicted_success'], cmap='RdYlGn', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Overlay actual restaurant locations
        plt.scatter(self.df_restaurants['longitude'], self.df_restaurants['latitude'], 
                   c='blue', s=20, alpha=0.5, label='Existing Restaurants')
        
        plt.colorbar(scatter, label='Predicted Success Score')
        plt.title('Restaurant Success Prediction Heat Map - Vancouver')
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
        # Derive primary cuisine
        if 'categories' in df.columns:
            df['primary_cuisine'] = df['categories'].fillna('').apply(lambda x: x.split(',')[0].strip() if x else '')
        elif 'cuisine' in df.columns:
            df['primary_cuisine'] = df['cuisine'].fillna('').astype(str)
        else:
            logger.warning("No categories/cuisine column; cannot derive cuisines.")
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
            props = {
                'cuisine': cuisine,
                'recommended_restaurant': top_row.get('name',''),
                'business_id': top_row.get('business_id',''),
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

    def run_complete_pipeline(self):
        """Run end-to-end pipeline including cuisine GeoJSON output."""
        print("VancouverPy: Restaurant Success Prediction Model Training")
        print("=" * 60)

        # 1. Load data
        if not self.load_processed_data():
            logger.error("Failed to load data. Run data collection/processing first.")
            return False

        # 2. Explore
        self.explore_data()

        # 3. Target
        self.prepare_target_variable()

        # 4. Feature importance
        self.analyze_feature_importance()

        # 5. Clustering
        cluster_labels, optimal_k = self.perform_clustering_analysis()
        if cluster_labels is None or len(cluster_labels) == 0:
            optimal_k = "N/A"

        # 6. Train models
        if not self.train_models():
            logger.error("Training failed.")
            return False

        # 7. Evaluate
        self.evaluate_models()

        # 8. Interpret best model
        self.interpret_best_model()

        # 9. Heat map
        self.create_prediction_heatmap()

        # 10. Save models
        self.save_models()

        # 11. Cuisine recommendation GeoJSON
        geo_path = self.generate_cuisine_recommendations_geojson()

        print("\nPipeline completed.")
        print(f"Best model: {self.best_model_name}")
        print(f"Optimal clusters: {optimal_k}")
        if geo_path:
            print(f"Cuisine recommendations written to: {geo_path}")
        return True


def main():
    """Main execution function"""
    predictor = RestaurantSuccessPredictor()
    success = predictor.run_complete_pipeline()
    if success:
        print("\nModel training completed successfully!")
        print("You can now use the trained models for restaurant success prediction.")
    else:
        print("\nModel training failed. Please check the logs for details.")


if __name__ == "__main__":
    main()
