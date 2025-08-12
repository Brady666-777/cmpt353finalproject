"""
PySpark Data Processing Script for VancouverPy

This script uses PySpark to efficiently process large datasets:
- business-licences.geojson (Vancouver business licenses)
- yelp_academic_dataset_business.json (Yelp business data - large file)
- yelp_academic_dataset_review.json (Yelp reviews - very large file)
- CensusProfile2021-ProfilRecensement2021-20250811051126.csv (Census data)

Author: VancouverPy Project Team
Date: August 2025
"""

import os
import builtins
import pandas as pd
import geopandas as gpd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded lightweight sentiment analyzer (previously imported)
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore
    SPARK_SA_TRANSFORMERS = True
except Exception:
    SPARK_SA_TRANSFORMERS = False

class MultilinguaSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if (SPARK_SA_TRANSFORMERS and 'torch' in globals() and torch.cuda.is_available()) else 'cpu'
        if SPARK_SA_TRANSFORMERS:
            self._load()
    def _load(self):
        for name in ["cardiffnlp/twitter-xlm-roberta-base-sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest"]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name)
                self.model.to(self.device)
                return
            except Exception:
                continue
        self.model = None
        self.tokenizer = None
    def predict(self, texts, batch_size=16):
        if not SPARK_SA_TRANSFORMERS or self.model is None:
            return self._fallback(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                enc = self.tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=256)
                enc = {k:v.to(self.device) for k,v in enc.items()}
                with torch.no_grad():
                    logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                for t,p in zip(batch, probs):
                    cl = int(torch.argmax(p).item())
                    conf = float(torch.max(p).item())
                    label = {0:'Negative',1:'Neutral',2:'Positive'}.get(cl,'Neutral')
                    out.append((label, conf, self._score(label, conf)))
            except Exception:
                out.extend([(l,0.6,s) for l,s in self._fallback_labels(batch)])
        return out
    def _score(self,label,conf):
        base={'Negative':0.2,'Neutral':0.5,'Positive':0.8}.get(label,0.5)
        if label=='Positive': return min(1.0, base + (conf-0.5)*0.4)
        if label=='Negative': return max(0.0, base - (conf-0.5)*0.4)
        return base
    def _fallback_labels(self,texts):
        pos=['good','great','excellent','amazing','love','best','awesome','fantastic','wonderful','perfect','delicious','outstanding']
        neg=['bad','terrible','awful','hate','worst','horrible','disgusting','disappointing','poor','rude','slow','expensive']
        res=[]
        for t in texts:
            tl=(t or '').lower()
            p=builtins.sum(1 for w in pos if w in tl)
            n=builtins.sum(1 for w in neg if w in tl)
            if p>n:
                res.append(('Positive',0.7))
            elif n>p:
                res.append(('Negative',0.3))
            else:
                res.append(('Neutral',0.5))
        return res
    def _fallback(self,texts):
        return [(l,0.6,s) for l,s in self._fallback_labels(texts)]

class SparkDataProcessor:
    """PySpark-based data processor for large datasets"""
    
    def __init__(self):
        # Use absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = os.path.join(base_dir, 'data', 'raw')
        self.processed_data_dir = os.path.join(base_dir, 'data', 'processed')
        
        # Ensure directories exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize Spark session
        self.spark = self._init_spark()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = MultilinguaSentimentAnalyzer()
        
    def _init_spark(self) -> SparkSession:
        """Initialize Spark session with optimized configuration"""
        logger.info("Initializing Spark session...")
        
        spark = SparkSession.builder \
            .appName("VancouverPy Data Processing") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Spark session initialized with version {spark.version}")
        return spark
    
    def load_business_licenses(self):
        """Load Vancouver business licenses using pandas (smaller dataset)"""
        logger.info("Loading business licenses data...")
        
        try:
            # Load with geopandas for geospatial data
            filepath = os.path.join(self.raw_data_dir, 'business-licences.geojson')
            gdf = gpd.read_file(filepath)
            
            logger.info(f"Loaded {len(gdf)} business license records")
            logger.info(f"Columns: {list(gdf.columns)}")
            
            # Filter for food-related businesses
            food_keywords = ['food', 'restaurant', 'cafe', 'bakery', 'catering', 'bar', 'pub', 'brewery']
            food_mask = gdf['businesstype'].str.lower().str.contains('|'.join(food_keywords), na=False)
            gdf_food = gdf[food_mask].copy()
            
            logger.info(f"Filtered to {len(gdf_food)} food-related businesses")
            
            # Filter for active businesses
            gdf_active = gdf_food[gdf_food['status'] == 'Issued'].copy()
            logger.info(f"Filtered to {len(gdf_active)} active businesses")
            
            # Extract coordinates
            gdf_active['latitude'] = gdf_active.geometry.y
            gdf_active['longitude'] = gdf_active.geometry.x
            
            # Save processed data
            output_path = os.path.join(self.processed_data_dir, 'business_licenses_processed.csv')
            gdf_active.drop('geometry', axis=1).to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            return gdf_active
            
        except Exception as e:
            logger.error(f"Error loading business licenses: {e}")
            return None
    
    def load_yelp_data_spark(self):
        """Load and process Yelp data using PySpark for efficiency"""
        logger.info("Loading Yelp business data with PySpark...")
        
        try:
            # Load JSON data
            filepath = os.path.join(self.raw_data_dir, 'yelp_academic_dataset_business.json')
            df = self.spark.read.json(filepath)
            
            logger.info(f"Loaded Yelp data with {df.count()} businesses")
            logger.info(f"Schema: {df.printSchema()}")
            
            # Filter for restaurants
            restaurant_filter = (
                col('categories').isNotNull() & 
                (lower(col('categories')).contains('restaurant') | 
                 lower(col('categories')).contains('food'))
            )
            
            df_restaurants = df.filter(restaurant_filter)
            restaurant_count = df_restaurants.count()
            logger.info(f"Found {restaurant_count} restaurants")
            
            # Filter for Canadian businesses (broader filter)
            canadian_filter = (
                (col('state').isin(['BC', 'AB', 'ON', 'QC', 'MB', 'SK', 'NB', 'NS', 'PE', 'NL', 'YT', 'NT', 'NU'])) |
                (lower(col('city')).contains('vancouver')) |
                (lower(col('city')).contains('toronto')) |
                (lower(col('city')).contains('montreal')) |
                (lower(col('city')).contains('calgary')) |
                (lower(col('city')).contains('ottawa')) |
                (lower(col('city')).contains('edmonton'))
            )
            
            df_canadian = df_restaurants.filter(canadian_filter)
            canadian_count = df_canadian.count()
            logger.info(f"Found {canadian_count} Canadian restaurants")
            
            # Focus on BC/Vancouver area
            bc_filter = (
                (col('state') == 'BC') |
                (lower(col('city')).contains('vancouver')) |
                (lower(col('city')).contains('burnaby')) |
                (lower(col('city')).contains('richmond')) |
                (lower(col('city')).contains('surrey'))
            )
            
            df_bc = df_canadian.filter(bc_filter)
            bc_count = df_bc.count()
            logger.info(f"Found {bc_count} BC/Vancouver area restaurants")
            
            # Convert to Pandas for easier manipulation
            df_pandas = df_bc.toPandas()
            
            if len(df_pandas) == 0:
                logger.warning("No BC restaurants found, using all Canadian restaurants")
                df_pandas = df_canadian.limit(1000).toPandas()  # Limit to prevent memory issues
            
            logger.info(f"Final dataset: {len(df_pandas)} restaurants")
            return df_pandas
            
        except Exception as e:
            logger.error(f"Error loading Yelp data: {e}")
            return pd.DataFrame()
    
    def load_yelp_reviews_spark(self, business_ids: List[str]):
        """Load Yelp reviews for specific businesses using PySpark"""
        if not business_ids:
            return pd.DataFrame()
            
        logger.info(f"Loading Yelp reviews for {len(business_ids)} businesses...")
        
        try:
            filepath = os.path.join(self.raw_data_dir, 'yelp_academic_dataset_review.json')
            
            # Load reviews
            df = self.spark.read.json(filepath)
            
            # Filter for our businesses
            df_filtered = df.filter(col('business_id').isin(business_ids))
            
            review_count = df_filtered.count()
            logger.info(f"Found {review_count} reviews for our businesses")
            
            # Convert to Pandas
            df_pandas = df_filtered.toPandas()
            
            return df_pandas
            
        except Exception as e:
            logger.error(f"Error loading Yelp reviews: {e}")
            return pd.DataFrame()
    
    def process_sentiment_analysis(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis to reviews"""
        if reviews_df.empty:
            return reviews_df
            
        logger.info(f"Processing sentiment analysis for {len(reviews_df)} reviews...")
        texts = reviews_df['text'].fillna('').astype(str).tolist()
        preds = self.sentiment_analyzer.predict(texts)
        sentiments = [p[0] for p in preds]
        confidences = [p[1] for p in preds]
        scores = [p[2] for p in preds]
        out = reviews_df.copy()
        out['sentiment'] = sentiments
        out['sentiment_confidence'] = confidences
        out['sentiment_score'] = scores
        logger.info("Sentiment analysis complete (embedded analyzer).")
        return out
    
    def aggregate_review_sentiment(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment features by business"""
        if reviews_df.empty:
            return pd.DataFrame()
            
        logger.info("Aggregating sentiment features by business...")
        
        # Group by business and calculate sentiment aggregates
        sentiment_features = reviews_df.groupby('business_id').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_confidence': 'mean',
            'stars': 'mean'
        }).round(4)
        
        # Flatten column names
        sentiment_features.columns = [
            'avg_sentiment_score',
            'sentiment_score_std',
            'sentiment_review_count',
            'avg_sentiment_confidence',
            'avg_review_rating'
        ]
        
        # Add sentiment distribution
        sentiment_dist = reviews_df.groupby(['business_id', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate percentages
        for col in sentiment_dist.columns:
            sentiment_dist[f'{col.lower()}_pct'] = (
                sentiment_dist[col] / sentiment_dist.sum(axis=1) * 100
            ).round(2)
        
        # Combine features
        sentiment_features = sentiment_features.join(sentiment_dist, how='left')
        sentiment_features = sentiment_features.fillna(0)
        
        # Reset index to make business_id a column
        sentiment_features = sentiment_features.reset_index()
        
        logger.info(f"Created sentiment features for {len(sentiment_features)} businesses")
        
        return sentiment_features
    
    def process_yelp_data(self, df_yelp: pd.DataFrame) -> pd.DataFrame:
        """Process and clean Yelp data"""
        if df_yelp.empty:
            return df_yelp
            
        logger.info("Processing Yelp data...")
        
        # Convert numeric columns
        numeric_cols = ['latitude', 'longitude', 'stars', 'review_count']
        for col in numeric_cols:
            if col in df_yelp.columns:
                df_yelp[col] = pd.to_numeric(df_yelp[col], errors='coerce')
        
        # Filter by valid coordinates (Vancouver area: roughly 49.0-49.4 lat, -123.3 to -122.8 lon)
        coord_filter = (
            (df_yelp['latitude'] >= 49.0) & (df_yelp['latitude'] <= 49.4) &
            (df_yelp['longitude'] >= -123.3) & (df_yelp['longitude'] <= -122.8)
        )
        
        df_filtered = df_yelp[coord_filter].copy()
        logger.info(f"Filtered to {len(df_filtered)} restaurants in Vancouver coordinates")
        
        if len(df_filtered) == 0:
            logger.warning("No restaurants in Vancouver coordinates, using broader filter")
            # Use BC coordinates instead
            bc_filter = (
                (df_yelp['latitude'] >= 48.0) & (df_yelp['latitude'] <= 60.0) &
                (df_yelp['longitude'] >= -140.0) & (df_yelp['longitude'] <= -110.0)
            )
            df_filtered = df_yelp[bc_filter].copy()
            logger.info(f"Using BC coordinates: {len(df_filtered)} restaurants")
        
        # Extract price level from attributes
        def extract_price_level(attributes_str):
            if pd.isna(attributes_str) or attributes_str is None:
                return 1
            try:
                if 'RestaurantsPriceRange2' in str(attributes_str):
                    import re
                    match = re.search(r"'RestaurantsPriceRange2':\s*'?(\d+)'?", str(attributes_str))
                    if match:
                        return int(match.group(1))
            except:
                pass
            return 1
        
        df_filtered['price_level'] = df_filtered['attributes'].apply(extract_price_level)
        
        # Fill missing values
        df_filtered['stars'].fillna(df_filtered['stars'].median(), inplace=True)
        df_filtered['review_count'].fillna(0, inplace=True)
        
        logger.info(f"Processed Yelp data: {len(df_filtered)} restaurants")
        
        return df_filtered
    
    def create_success_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create success score based on ratings and reviews"""
        if df.empty:
            return df
            
        logger.info("Creating Success Score...")
        
        # Normalize components
        stars_norm = df['stars'] / 5.0  # 0-1 scale
        
        # Log transform review count to handle skewness
        review_log = np.log1p(df['review_count'])
        review_norm = (review_log - review_log.min()) / (review_log.max() - review_log.min() + 1e-8)
        
        # Combine with weights
        df['success_score'] = (0.6 * stars_norm + 0.4 * review_norm)
        
        logger.info(f"Success Score created. Mean: {df['success_score'].mean():.3f}, Std: {df['success_score'].std():.3f}")
        
        return df
    
    def calculate_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate competitive features using efficient methods"""
        if df.empty or len(df) < 2:
            logger.warning("Insufficient data for competitive analysis")
            df['restaurants_within_500m'] = 0
            df['avg_competitor_rating'] = df.get('stars', 0)
            return df
            
        logger.info("Calculating competitive features...")
        
        from sklearn.neighbors import BallTree
        
        # Prepare coordinates
        coords = df[['latitude', 'longitude']].dropna()
        
        if len(coords) < 2:
            logger.warning("No valid coordinates found for competitive analysis")
            df['restaurants_within_500m'] = 0
            df['avg_competitor_rating'] = df.get('stars', 0)
            return df
        
        # Create BallTree for efficient distance calculations
        tree = BallTree(np.radians(coords[['latitude', 'longitude']].values), metric='haversine')
        
        # Find neighbors within 500m (0.5km)
        earth_radius = 6371000  # meters
        radius = 0.5 / earth_radius  # 500m in radians
        
        restaurants_within_500m = []
        avg_competitor_ratings = []
        
        for idx, (lat, lon) in coords.iterrows():
            # Query neighbors
            indices = tree.query_radius([[np.radians(lat), np.radians(lon)]], r=radius)[0]
            
            # Exclude self
            neighbor_indices = [i for i in indices if coords.index[i] != idx]
            
            restaurants_within_500m.append(len(neighbor_indices))
            
            if neighbor_indices:
                neighbor_ratings = df.loc[coords.index[neighbor_indices], 'stars'].dropna()
                avg_competitor_ratings.append(neighbor_ratings.mean() if len(neighbor_ratings) > 0 else df['stars'].mean())
            else:
                avg_competitor_ratings.append(df['stars'].mean())
        
        # Map back to original dataframe
        df['restaurants_within_500m'] = 0
        df['avg_competitor_rating'] = df['stars'].mean()
        
        df.loc[coords.index, 'restaurants_within_500m'] = restaurants_within_500m
        df.loc[coords.index, 'avg_competitor_rating'] = avg_competitor_ratings
        
        logger.info(f"Competitive features calculated for {len(coords)} restaurants")
        
        return df
    
    def prepare_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final features for modeling"""
        logger.info("Preparing model features...")
        
        # Select key features including sentiment
        feature_columns = [
            'latitude', 'longitude', 'stars', 'review_count', 'price_level',
            'restaurants_within_500m', 'avg_competitor_rating', 'success_score',
            'avg_sentiment_score', 'sentiment_score_std', 'avg_sentiment_confidence',
            'positive_pct', 'negative_pct', 'neutral_pct'
        ]
        
        # Keep only available columns
        available_features = [col for col in feature_columns if col in df.columns]
        df_features = df[available_features].copy()
        
        # Handle missing values
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_columns] = df_features[numeric_columns].fillna(df_features[numeric_columns].median())
        
        logger.info(f"Model features prepared: {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        
        return df_features
    
    def process_all_data(self):
        """Main processing pipeline using PySpark for large datasets"""
        logger.info("Starting PySpark data processing pipeline...")

        try:
            # Process business licenses (smaller dataset - use pandas)
            business_licenses = self.load_business_licenses()

            # Process Yelp data (large dataset - use PySpark)
            yelp_data = self.load_yelp_data_spark()

            if not yelp_data.empty:
                # Process Yelp data
                yelp_processed = self.process_yelp_data(yelp_data)

                # Load reviews for our businesses
                if len(yelp_processed) > 0:
                    business_ids = yelp_processed['business_id'].tolist()
                    reviews = self.load_yelp_reviews_spark(business_ids)

                    if not reviews.empty:
                        # Process sentiment analysis
                        reviews_with_sentiment = self.process_sentiment_analysis(reviews)

                        # Aggregate sentiment features
                        sentiment_features = self.aggregate_review_sentiment(reviews_with_sentiment)

                        # Aggregate other review metrics
                        review_agg = reviews.groupby('business_id').agg({
                            'stars': ['mean', 'count'],
                            'useful': 'sum',
                            'funny': 'sum',
                            'cool': 'sum'
                        }).reset_index()

                        review_agg.columns = ['business_id', 'avg_review_stars', 'total_reviews',
                                              'total_useful', 'total_funny', 'total_cool']

                        # Merge with business data
                        yelp_processed = yelp_processed.merge(review_agg, on='business_id', how='left')

                        # Merge sentiment features
                        if not sentiment_features.empty:
                            yelp_processed = yelp_processed.merge(sentiment_features, on='business_id', how='left')
                            logger.info("Sentiment analysis features integrated successfully!")

                # Calculate competitive features
                yelp_processed = self.calculate_competitive_features(yelp_processed)

                # Create success score
                yelp_processed = self.create_success_score(yelp_processed)

                # Prepare model features
                model_features = self.prepare_model_features(yelp_processed)

                # Save processed data
                output_path = os.path.join(self.processed_data_dir, 'restaurants_with_features_spark.csv')
                yelp_processed.to_csv(output_path, index=False)
                logger.info(f"Saved processed data to {output_path}")

                features_path = os.path.join(self.processed_data_dir, 'model_features_spark.csv')
                model_features.to_csv(features_path, index=False)
                logger.info(f"Saved model features to {features_path}")

                # Print summary
                logger.info("PySpark data processing completed successfully!")
                print(f"""
PROCESSING SUMMARY:
Total restaurants processed: {len(yelp_processed)}
Features created: {len(model_features.columns)}
Feature names: {list(model_features.columns)}
Success score range: {model_features.get('success_score', pd.Series([0])).min():.3f} - {model_features.get('success_score', pd.Series([0])).max():.3f}
Average success score: {model_features.get('success_score', pd.Series([0])).mean():.3f}
                """)

            else:
                logger.error("No Yelp data could be loaded")

        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise

        finally:
            # Clean up Spark session
            if hasattr(self, 'spark'):
                self.spark.stop()
                logger.info("Spark session stopped")

def main():
    """Main execution function"""
    processor = SparkDataProcessor()
    processor.process_all_data()

if __name__ == "__main__":
    main()
