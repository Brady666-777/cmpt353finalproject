"""
Updated Data Processing Script for Real VancouverPy Data

This script processes the actual datasets provided:
- business-licences.geojson (Vancouver business licenses)
- CensusProfile2021-ProfilRecensement2021-20250811051126.csv (Census data)
- good-restaurant-in-vancouver-overview.csv (Google restaurant data)
- dataset_crawler-google-review_2025-08-06_03-55-37-484.csv (Google reviews data)

Includes inline sentiment analysis for enhanced feature engineering.

Author: VancouverPy Project Team
Date: August 2025
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import json  # Add json import for GeoJSON generation
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import re
import logging
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis imports
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InlineSentimentAnalyzer:
    """Inline sentiment analysis for restaurant data processing"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("Transformers not available. Using simple fallback sentiment analysis.")
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        try:
            # Try multilingual model first
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            logger.info(f"Loading sentiment model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            
            logger.info(f"Successfully loaded sentiment model on {self.device}")
            
        except Exception as e:
            logger.warning(f"Failed to load multilingual model: {e}")
            try:
                # Fallback to English-only model
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                logger.info(f"Loading fallback sentiment model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                
                logger.info(f"Successfully loaded fallback model on {self.device}")
                
            except Exception as e2:
                logger.error(f"Failed to load any sentiment model: {e2}")
                self.model = None
                self.tokenizer = None
    
    def analyze_text_sentiment(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for a list of texts"""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return self._fallback_sentiment(texts)
        
        results = []
        batch_size = 16
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, texts: List[str]) -> List[Dict]:
        """Process a batch of texts"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert to probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Map predictions to sentiment labels
            results = []
            for text, probs in zip(texts, probabilities):
                predicted_class = torch.argmax(probs).item()
                confidence = torch.max(probs).item()
                
                # Map class to sentiment
                sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                sentiment = sentiment_map.get(predicted_class, "Neutral")
                
                results.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'sentiment_score': self._convert_to_score(sentiment, confidence)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [{'sentiment': 'Neutral', 'confidence': 0.5, 'sentiment_score': 0.5} 
                   for _ in texts]
    
    def _convert_to_score(self, sentiment: str, confidence: float) -> float:
        """Convert sentiment label to numerical score (0-1)"""
        base_scores = {'Negative': 0.2, 'Neutral': 0.5, 'Positive': 0.8}
        base_score = base_scores.get(sentiment, 0.5)
        
        # Adjust based on confidence
        if sentiment == 'Positive':
            return min(1.0, base_score + (confidence - 0.5) * 0.4)
        elif sentiment == 'Negative':
            return max(0.0, base_score - (confidence - 0.5) * 0.4)
        else:
            return base_score
    
    def _fallback_sentiment(self, texts: List[str]) -> List[Dict]:
        """Simple fallback sentiment analysis using keyword matching"""
        logger.info("Using fallback sentiment analysis")
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 
                         'fantastic', 'wonderful', 'perfect', 'delicious', 'outstanding',
                         'brilliant', 'superb', 'magnificent', 'exceptional', 'incredible',
                         'fabulous', 'marvelous', 'spectacular', 'phenomenal']
        
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting',
                         'disappointing', 'poor', 'rude', 'slow', 'expensive', 'overpriced',
                         'mediocre', 'bland', 'tasteless', 'cold', 'dirty', 'unfriendly',
                         'unprofessional', 'crowded', 'noisy', 'uncomfortable']
        
        results = []
        for text in texts:
            text_lower = str(text).lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = 'Positive'
                score = min(0.9, 0.6 + positive_count * 0.1)
            elif negative_count > positive_count:
                sentiment = 'Negative'
                score = max(0.1, 0.4 - negative_count * 0.1)
            else:
                sentiment = 'Neutral'
                score = 0.5
            
            results.append({
                'sentiment': sentiment,
                'confidence': 0.6,
                'sentiment_score': score
            })
        
        return results

class RealDataProcessor:
    """Updated data processor for real datasets"""
    
    def __init__(self):
        # Use absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = os.path.join(base_dir, 'data', 'raw')
        self.processed_data_dir = os.path.join(base_dir, 'data', 'processed')
        
        # Ensure processed data directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = InlineSentimentAnalyzer()
        
    def load_business_licenses(self) -> gpd.GeoDataFrame:
        """Load and process Vancouver business licenses GeoJSON"""
        logger.info("Loading business licenses data...")
        
        try:
            # Load the GeoJSON file
            gdf = gpd.read_file(f"{self.raw_data_dir}/business-licences.geojson")
            
            logger.info(f"Loaded {len(gdf)} business license records")
            logger.info(f"Columns: {list(gdf.columns)}")
            
            # Filter for food-related businesses
            food_keywords = ['restaurant', 'cafe', 'bakery', 'food', 'coffee', 'bar', 'pub', 'deli', 'bistro']
            
            if 'businesstype' in gdf.columns:
                food_filter = gdf['businesstype'].str.lower().str.contains('|'.join(food_keywords), na=False)
                gdf_food = gdf[food_filter].copy()
                logger.info(f"Filtered to {len(gdf_food)} food-related businesses")
            else:
                gdf_food = gdf.copy()
                logger.warning("No 'businesstype' column found. Using all businesses.")
            
            # Filter for active businesses
            if 'status' in gdf_food.columns:
                active_filter = gdf_food['status'].str.lower().isin(['issued', 'active'])
                gdf_food = gdf_food[active_filter].copy()
                logger.info(f"Filtered to {len(gdf_food)} active businesses")
            
            return gdf_food
            
        except Exception as e:
            logger.error(f"Error loading business licenses: {e}")
            return gpd.GeoDataFrame()
    
    def load_google_restaurant_overview(self) -> pd.DataFrame:
        """Load and process Google restaurant overview data"""
        logger.info("Loading Google restaurant overview data...")
        
        try:
            df = pd.read_csv(f"{self.raw_data_dir}/good-restaurant-in-vancouver-overview.csv")
            
            logger.info(f"Loaded {len(df)} restaurants from Google overview")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Clean and standardize columns
            df_clean = df.copy()
            
            # Rename columns for consistency
            column_mapping = {
                'name': 'business_name',
                'rating': 'stars',
                'reviews': 'review_count',
                'main_category': 'primary_category',
                'address': 'full_address'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df_clean.columns:
                    df_clean[new_col] = df_clean[old_col]
            
            # Clean numeric columns
            if 'stars' in df_clean.columns:
                df_clean['stars'] = pd.to_numeric(df_clean['stars'], errors='coerce')
            if 'review_count' in df_clean.columns:
                df_clean['review_count'] = pd.to_numeric(df_clean['review_count'], errors='coerce')
            
            # Extract price level (default to moderate)
            df_clean['price_level'] = 2
            
            logger.info(f"Processed Google overview data: {len(df_clean)} restaurants")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error loading Google overview data: {e}")
            return pd.DataFrame()
    
    def load_google_reviews_data(self) -> pd.DataFrame:
        """Load and process Google reviews data"""
        logger.info("Loading Google reviews data...")
        
        try:
            df = pd.read_csv(f"{self.raw_data_dir}/dataset_crawler-google-review_2025-08-06_03-55-37-484.csv")
            
            logger.info(f"Loaded {len(df)} restaurants from Google reviews")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Clean and standardize columns
            df_clean = df.copy()
            
            # Rename columns for consistency
            column_mapping = {
                'title': 'business_name',
                'totalScore': 'stars',
                'reviewsCount': 'review_count',
                'categoryName': 'primary_category',
                'street': 'address'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df_clean.columns:
                    df_clean[new_col] = df_clean[old_col]
            
            # Create full address
            if 'address' in df_clean.columns and 'city' in df_clean.columns:
                df_clean['full_address'] = df_clean['address'] + ', ' + df_clean['city'] + ', BC'
            
            # Clean numeric columns
            if 'stars' in df_clean.columns:
                df_clean['stars'] = pd.to_numeric(df_clean['stars'], errors='coerce')
            if 'review_count' in df_clean.columns:
                df_clean['review_count'] = pd.to_numeric(df_clean['review_count'], errors='coerce')
            
            # Extract price level (default to moderate)
            df_clean['price_level'] = 2
            
            logger.info(f"Processed Google reviews data: {len(df_clean)} restaurants")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error loading Google reviews data: {e}")
            return pd.DataFrame()
    
    def geocode_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geocode addresses to get coordinates"""
        logger.info("Geocoding addresses...")
        
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        import time
        
        geolocator = Nominatim(user_agent="vancouvepy_restaurant_predictor")
        
        df_geo = df.copy()
        df_geo['latitude'] = None
        df_geo['longitude'] = None
        
        if 'full_address' not in df_geo.columns:
            logger.warning("No full_address column found for geocoding")
            return df_geo
        
        successful_geocodes = 0
        
        for idx, row in df_geo.iterrows():
            if idx % 20 == 0:
                logger.info(f"Geocoded {idx}/{len(df_geo)} addresses...")
            
            try:
                address = row['full_address']
                if pd.notna(address):
                    # Add rate limiting
                    time.sleep(1)
                    
                    location = geolocator.geocode(address, timeout=10)
                    if location:
                        df_geo.loc[idx, 'latitude'] = location.latitude
                        df_geo.loc[idx, 'longitude'] = location.longitude
                        successful_geocodes += 1
                    
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logger.warning(f"Geocoding failed for address {address}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error geocoding {address}: {e}")
                continue
        
        logger.info(f"Successfully geocoded {successful_geocodes}/{len(df_geo)} addresses")
        
        # Filter out rows without coordinates
        df_geo = df_geo.dropna(subset=['latitude', 'longitude'])
        
        # Filter for Vancouver area coordinates
        vancouver_lat_range = (49.0, 49.4)
        vancouver_lon_range = (-123.5, -122.8)
        
        coord_filter = (
            (df_geo['latitude'] >= vancouver_lat_range[0]) & 
            (df_geo['latitude'] <= vancouver_lat_range[1]) &
            (df_geo['longitude'] >= vancouver_lon_range[0]) & 
            (df_geo['longitude'] <= vancouver_lon_range[1])
        )
        
        df_geo = df_geo[coord_filter].copy()
        logger.info(f"Filtered to {len(df_geo)} restaurants in Vancouver coordinates")
        
        return df_geo
    
    def load_census_data(self) -> pd.DataFrame:
        """Load and process census data"""
        logger.info("Loading census data...")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df_census = None
            
            filepath = f"{self.raw_data_dir}/CensusProfile2021-ProfilRecensement2021-20250811051126.csv"
            
            for encoding in encodings:
                try:
                    df_census = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"Successfully loaded census data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df_census is None:
                logger.error("Failed to load census data with any common encoding")
                return pd.DataFrame()
            
            logger.info(f"Loaded census data: {df_census.shape}")
            logger.info(f"Census columns: {list(df_census.columns)[:10]}")
            
            return df_census
            
        except Exception as e:
            logger.error(f"Error loading census data: {e}")
            return pd.DataFrame()
    
    def process_business_data(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Clean and process business license data"""
        logger.info("Processing business license data...")
        
        if gdf.empty:
            return pd.DataFrame()
        
        # Convert to regular DataFrame
        df = gdf.copy()
        
        # Extract coordinates from geometry
        if 'geometry' in df.columns:
            df['latitude'] = df.geometry.y
            df['longitude'] = df.geometry.x
        
        # Extract key features
        essential_columns = ['businessname', 'businesstype', 'status', 'localarea', 
                           'latitude', 'longitude', 'street', 'city']
        
        # Keep only available essential columns
        available_columns = [col for col in essential_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        
        # Clean coordinates
        if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['latitude', 'longitude'])
            logger.info(f"Cleaned coordinates for {len(df_clean)} businesses")
        
        # Add default values for modeling
        df_clean['price_level'] = 2  # Default moderate pricing
        df_clean['review_count'] = 10  # Default review count
        df_clean['stars'] = 4.0  # Default rating
        
        # Extract primary category from business type
        if 'businesstype' in df_clean.columns:
            df_clean['primary_category'] = df_clean['businesstype'].apply(
                lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Restaurant'
            )
        else:
            df_clean['primary_category'] = 'Restaurant'
        
        # Add sentiment analysis for business descriptions
        logger.info("Analyzing sentiment for business descriptions...")
        self._add_sentiment_features(df_clean)
        
        logger.info(f"Processed business data: {len(df_clean)} businesses")
        return df_clean
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> None:
        """Add sentiment analysis features to the dataframe"""
        try:
            # Prepare text data for sentiment analysis
            text_data = []
            
            # Combine available text fields for sentiment analysis
            for _, row in df.iterrows():
                text_parts = []
                
                # Add business name
                if pd.notna(row.get('businessname')):
                    text_parts.append(str(row['businessname']))
                
                # Add business type
                if pd.notna(row.get('businesstype')):
                    text_parts.append(str(row['businesstype']))
                
                # Add location area (can indicate neighborhood quality)
                if pd.notna(row.get('localarea')):
                    text_parts.append(str(row['localarea']))
                
                # Combine text or use default
                combined_text = ' '.join(text_parts) if text_parts else 'Restaurant'
                text_data.append(combined_text)
            
            # Analyze sentiment
            if text_data:
                logger.info(f"Analyzing sentiment for {len(text_data)} business descriptions...")
                sentiment_results = self.sentiment_analyzer.analyze_text_sentiment(text_data)
                
                # Add sentiment features to dataframe
                df['sentiment_score'] = [result['sentiment_score'] for result in sentiment_results]
                df['sentiment_label'] = [result['sentiment'] for result in sentiment_results]
                df['sentiment_confidence'] = [result['confidence'] for result in sentiment_results]
                
                logger.info(f"Added sentiment features. Average sentiment score: {df['sentiment_score'].mean():.3f}")
            else:
                # Default values if no text available
                df['sentiment_score'] = 0.5
                df['sentiment_label'] = 'Neutral'
                df['sentiment_confidence'] = 0.5
                logger.warning("No text data available for sentiment analysis, using defaults")
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Add default sentiment features
            df['sentiment_score'] = 0.5
            df['sentiment_label'] = 'Neutral'
            df['sentiment_confidence'] = 0.5
    
    def calculate_competitive_features(self, df: pd.DataFrame, radius_km: float = 0.5) -> pd.DataFrame:
        """Calculate competitive density features using optimized approach"""
        logger.info(f"Calculating competitive features within {radius_km}km...")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.warning("No coordinates available for competitive analysis")
            return df
        
        df_comp = df.copy()
        
        # Filter out rows with NaN coordinates
        df_comp = df_comp.dropna(subset=['latitude', 'longitude'])
        
        if len(df_comp) == 0:
            logger.warning("No valid coordinates found for competitive analysis")
            return df
        
        # For efficiency, use simplified competitive analysis for large datasets
        if len(df_comp) > 1000:
            logger.info("Large dataset detected. Using simplified competitive analysis...")
            
            # Create grid-based density approximation
            lat_bins = pd.cut(df_comp['latitude'], bins=20)
            lon_bins = pd.cut(df_comp['longitude'], bins=20)
            
            # Count businesses in each grid cell
            grid_counts = df_comp.groupby([lat_bins, lon_bins]).size()
            
            # Assign density based on grid cell
            df_comp['competitor_count'] = 0
            df_comp['similar_cuisine_count'] = 0
            
            for idx, row in df_comp.iterrows():
                lat_bin = pd.cut([row['latitude']], bins=lat_bins.cat.categories)[0]
                lon_bin = pd.cut([row['longitude']], bins=lon_bins.cat.categories)[0]
                
                # Get count for this grid cell
                try:
                    cell_count = grid_counts.get((lat_bin, lon_bin), 1)
                    df_comp.loc[idx, 'competitor_count'] = max(0, cell_count - 1)  # Exclude self
                    df_comp.loc[idx, 'similar_cuisine_count'] = max(0, cell_count // 2)  # Approximate
                except:
                    df_comp.loc[idx, 'competitor_count'] = 5  # Default value
                    df_comp.loc[idx, 'similar_cuisine_count'] = 2
            
            logger.info("Grid-based competitive analysis complete")
            
        else:
            # Original detailed analysis for smaller datasets
            df_comp['competitor_count'] = 0
            df_comp['similar_cuisine_count'] = 0
            
            # Calculate for each restaurant
            for idx, row in df_comp.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processed {idx}/{len(df_comp)} restaurants...")
                
                try:
                    restaurant_coords = (row['latitude'], row['longitude'])
                    
                    # Skip if coordinates are invalid
                    if pd.isna(restaurant_coords[0]) or pd.isna(restaurant_coords[1]):
                        continue
                        
                    competitors = 0
                    similar_cuisine = 0
                    
                    # Check distance to all other restaurants
                    for other_idx, other_row in df_comp.iterrows():
                        if idx != other_idx:
                            try:
                                other_coords = (other_row['latitude'], other_row['longitude'])
                                
                                # Skip if other coordinates are invalid
                                if pd.isna(other_coords[0]) or pd.isna(other_coords[1]):
                                    continue
                                    
                                distance = geodesic(restaurant_coords, other_coords).kilometers
                                
                                if distance <= radius_km:
                                    competitors += 1
                                    
                                    # Check if similar cuisine
                                    if ('primary_category' in df_comp.columns and 
                                        row.get('primary_category') == other_row.get('primary_category')):
                                        similar_cuisine += 1
                            except:
                                continue
                    
                    df_comp.loc[idx, 'competitor_count'] = competitors
                    df_comp.loc[idx, 'similar_cuisine_count'] = similar_cuisine
                    
                except Exception as e:
                    logger.warning(f"Error processing restaurant {idx}: {e}")
                    continue
            
            logger.info("Detailed competitive features calculation complete")
        
        return df_comp
    
    def create_success_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: Success Score"""
        logger.info("Creating Success Score...")
        
        df_target = df.copy()
        
        # Components of success score
        # 1. Rating (normalized to 0-1)
        if 'stars' in df_target.columns:
            rating_normalized = (df_target['stars'] - 1) / 4  # Yelp ratings are 1-5
            rating_normalized = rating_normalized.fillna(0.5)
        else:
            rating_normalized = 0.5
        
        # 2. Review count (log-normalized)
        if 'review_count' in df_target.columns:
            # Handle zero review counts
            review_count_safe = df_target['review_count'].fillna(1).clip(lower=1)
            review_log = np.log1p(review_count_safe)
            review_normalized = (review_log - review_log.min()) / (review_log.max() - review_log.min())
            review_normalized = review_normalized.fillna(0.5)
        else:
            review_normalized = 0.5
        
        # 3. Business longevity (placeholder - would need opening date)
        longevity_score = 0.5  # Default value
        
        # Weighted combination
        weights = {'rating': 0.4, 'reviews': 0.4, 'longevity': 0.2}
        
        df_target['success_score'] = (
            weights['rating'] * rating_normalized +
            weights['reviews'] * review_normalized +
            weights['longevity'] * longevity_score
        )
        
        logger.info(f"Success Score created. Mean: {df_target['success_score'].mean():.3f}, "
                   f"Std: {df_target['success_score'].std():.3f}")
        
        return df_target
    
    def prepare_model_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for machine learning"""
        logger.info("Preparing model features...")
        
        # Define feature columns that we want to use
        potential_features = [
            'latitude', 'longitude', 'stars', 'review_count', 'price_level',
            'competitor_count', 'similar_cuisine_count',
            'sentiment_score', 'sentiment_confidence'  # Add sentiment features
        ]
        
        # Keep only available features
        available_features = [col for col in potential_features if col in df.columns]
        
        if not available_features:
            logger.error("No suitable features found for modeling")
            return pd.DataFrame(), []
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode categorical variables if any
        categorical_cols = X.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
        
        logger.info(f"Model features prepared: {len(available_features)} features")
        logger.info(f"Features: {available_features}")
        
        return X, available_features
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        filepath = f"{self.processed_data_dir}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def process_all_data(self):
        """Main processing pipeline for real data"""
        logger.info("Starting real data processing pipeline...")
        
        # Step 1: Load and process business licenses (these already have coordinates)
        business_licenses = self.load_business_licenses()
        if not business_licenses.empty:
            business_data = self.process_business_data(business_licenses)
            logger.info(f"Processed {len(business_data)} business license records")
            
            if len(business_data) > 0:
                # Calculate competitive features
                business_with_competition = self.calculate_competitive_features(business_data)

                # Create success score
                business_final = self.create_success_score(business_with_competition)

                # Prepare model features
                X, feature_names = self.prepare_model_features(business_final)

                if not X.empty:
                    # Save processed data
                    self.save_processed_data(business_final, 'restaurants_with_features.csv')
                    self.save_processed_data(X, 'model_features.csv')

                    # Save feature names
                    pd.Series(feature_names).to_csv(
                        f"{self.processed_data_dir}/feature_names.csv",
                        index=False, header=['feature']
                    )

                    logger.info("Data processing completed successfully!")
                    print(f"\nPROCESSING SUMMARY:")
                    print(f"Total restaurants processed: {len(business_final)}")
                    print(f"Features created: {len(feature_names)}")
                    print(f"Feature names: {feature_names}")
                    print(f"Success score range: {business_final['success_score'].min():.3f} - {business_final['success_score'].max():.3f}")
                    print(f"Average success score: {business_final['success_score'].mean():.3f}")

                    # Save Google data separately for reference (without geocoding)
                    self.save_google_data_for_reference()
                    
                    # Generate cuisine-specific recommendations
                    self.generate_cuisine_recommendations(business_final)

                else:
                    logger.error("No features could be created for modeling")
            else:
                logger.error("No restaurants found for processing")
        else:
            logger.error("No business license data could be loaded")

        # Step 2: Load census data (for reference)
        census_data = self.load_census_data()
        if not census_data.empty:
            self.save_processed_data(census_data, 'census_data_raw.csv')
    
    def save_google_data_for_reference(self):
        """Save Google data for reference without geocoding"""
        logger.info("Saving Google data for reference...")
        
        try:
            # Load and save Google overview data
            google_overview = self.load_google_restaurant_overview()
            if not google_overview.empty:
                self.save_processed_data(google_overview, 'google_restaurants_overview.csv')
                logger.info(f"Saved {len(google_overview)} Google overview restaurants")
            
            # Load and save Google reviews data
            google_reviews = self.load_google_reviews_data()
            if not google_reviews.empty:
                self.save_processed_data(google_reviews, 'google_restaurants_reviews.csv')
                logger.info(f"Saved {len(google_reviews)} Google reviews restaurants")
                
        except Exception as e:
            logger.error(f"Error saving Google data: {e}")
    
    def generate_cuisine_recommendations(self, restaurant_data: pd.DataFrame):
        """Generate optimal location recommendations for each cuisine type as GeoJSON"""
        logger.info("Generating cuisine-specific location recommendations...")
        
        try:
            # Load Google data to get cuisine information
            google_overview = self.load_google_restaurant_overview()
            
            # Define cuisine types from Google data and common cuisines
            cuisine_types = [
                'Italian restaurant', 'Indian restaurant', 'French restaurant', 
                'Seafood restaurant', 'Chinese restaurant', 'Japanese restaurant',
                'Mexican restaurant', 'Thai restaurant', 'Korean restaurant',
                'Middle Eastern restaurant', 'Spanish restaurant', 
                'Pacific Northwest restaurant (Canada)', 'Canadian restaurant',
                'Steak house', 'Fine dining restaurant', 'Breakfast restaurant',
                'Ramen restaurant', 'Peruvian restaurant', 'Vegan restaurant',
                'Asian restaurant', 'Vietnamese restaurant', 'Greek restaurant',
                'Turkish restaurant', 'Lebanese restaurant', 'Sushi restaurant'
            ]
            
            # Create grid of potential locations across Vancouver
            lat_range = np.linspace(49.204, 49.298, 20)  # Vancouver latitude range
            lon_range = np.linspace(-123.211, -123.024, 25)  # Vancouver longitude range
            
            recommendations = []
            
            for cuisine in cuisine_types:
                logger.info(f"Analyzing optimal locations for {cuisine}...")
                
                # Calculate optimal location for this cuisine
                optimal_locations = self._find_optimal_locations_for_cuisine(
                    restaurant_data, cuisine, lat_range, lon_range, top_n=3
                )
                
                for i, location in enumerate(optimal_locations):
                    recommendation = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [float(location['longitude']), float(location['latitude'])]
                        },
                        'properties': {
                            'cuisine_type': cuisine,
                            'rank': int(i + 1),
                            'predicted_success_score': float(location['success_score']),
                            'competitor_count': int(location['competitor_count']),
                            'similar_cuisine_count': int(location['similar_cuisine_count']),
                            'local_area': location.get('neighborhood', 'Vancouver'),
                            'neighborhood': location.get('neighborhood', 'Central Vancouver'),
                            'recommendation_reason': location['reason'],
                            'confidence': float(location['confidence'])
                        }
                    }
                    recommendations.append(recommendation)
            
            # Create GeoJSON structure
            geojson_data = {
                'type': 'FeatureCollection',
                'features': recommendations,
                'properties': {
                    'title': 'Vancouver Restaurant Location Recommendations by Cuisine',
                    'description': 'AI-generated optimal locations for new restaurants by cuisine type',
                    'generated_date': pd.Timestamp.now().isoformat(),
                    'total_recommendations': len(recommendations),
                    'cuisine_types_analyzed': len(cuisine_types)
                }
            }
            
            # Save as GeoJSON
            output_path = f"{self.processed_data_dir}/recommended_spots.geojson"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated {len(recommendations)} cuisine recommendations")
            logger.info(f"Recommendations saved to {output_path}")
            
            # Create summary report
            self._create_recommendation_summary(cuisine_types, recommendations)
            
        except Exception as e:
            logger.error(f"Error generating cuisine recommendations: {e}")
    
    def _find_optimal_locations_for_cuisine(self, restaurant_data: pd.DataFrame, 
                                           cuisine: str, lat_range: np.ndarray, 
                                           lon_range: np.ndarray, top_n: int = 3) -> List[Dict]:
        """Find optimal locations for a specific cuisine type using realistic analysis"""
        
        locations = []
        cuisine_preferences = self._get_cuisine_preferences(cuisine)
        
        # Define distinct Vancouver neighborhoods with realistic coordinates and variations
        neighborhood_centers = [
            {"name": "Downtown", "lat": 49.2827, "lon": -123.1207},
            {"name": "West End", "lat": 49.2915, "lon": -123.1348},
            {"name": "Yaletown", "lat": 49.2745, "lon": -123.1210},
            {"name": "Gastown", "lat": 49.2838, "lon": -123.1085},
            {"name": "Chinatown", "lat": 49.2808, "lon": -123.1025},
            {"name": "Mount Pleasant", "lat": 49.2622, "lon": -123.1011},
            {"name": "Commercial Drive", "lat": 49.2650, "lon": -123.0692},
            {"name": "Kitsilano", "lat": 49.2688, "lon": -123.1533},
            {"name": "South Granville", "lat": 49.2448, "lon": -123.1365},
            {"name": "Main Street", "lat": 49.2520, "lon": -123.1003},
            {"name": "Robson Street", "lat": 49.2842, "lon": -123.1189},
            {"name": "Davie Village", "lat": 49.2806, "lon": -123.1396},
            {"name": "Olympic Village", "lat": 49.2667, "lon": -123.1135},
            {"name": "Coal Harbour", "lat": 49.2906, "lon": -123.1226},
            {"name": "Fairview", "lat": 49.2591, "lon": -123.1266},
            {"name": "Riley Park", "lat": 49.2485, "lon": -123.1046},
            {"name": "Hastings-Sunrise", "lat": 49.2789, "lon": -123.0459},
            {"name": "Strathcona", "lat": 49.2735, "lon": -123.0893},
            {"name": "Grandview-Woodland", "lat": 49.2700, "lon": -123.0700},
            {"name": "Kensington-Cedar Cottage", "lat": 49.2389, "lon": -123.0764}
        ]
        
        # Generate candidate locations with much more variation
        np.random.seed(hash(cuisine) % 1000)  # Different seed per cuisine for variety
        
        # Select different neighborhood preferences based on cuisine type
        cuisine_neighborhood_preferences = {
            'Italian restaurant': ['Commercial Drive', 'Mount Pleasant', 'Kitsilano'],
            'Chinese restaurant': ['Chinatown', 'Main Street', 'Richmond'],
            'Japanese restaurant': ['Robson Street', 'West End', 'Kitsilano'],
            'Thai restaurant': ['Commercial Drive', 'Main Street', 'Fairview'],
            'Indian restaurant': ['Main Street', 'Commercial Drive', 'Sunset'],
            'French restaurant': ['Yaletown', 'Gastown', 'South Granville'],
            'Korean restaurant': ['Robson Street', 'West End', 'Burnaby'],
            'Vietnamese restaurant': ['Commercial Drive', 'Main Street', 'Kingsway'],
            'Mexican restaurant': ['Commercial Drive', 'Kitsilano', 'Mount Pleasant'],
            'Fine dining restaurant': ['Yaletown', 'West End', 'Coal Harbour'],
            'Middle Eastern restaurant': ['Commercial Drive', 'Main Street', 'North Van'],
            'Breakfast restaurant': ['Kitsilano', 'Mount Pleasant', 'Commercial Drive']
        }
        
        preferred_neighborhoods = cuisine_neighborhood_preferences.get(cuisine, 
            ['Downtown', 'Commercial Drive', 'Main Street'])
        
        # Filter neighborhoods to preferred ones, with fallback to all
        available_neighborhoods = [n for n in neighborhood_centers 
                                 if n['name'] in preferred_neighborhoods]
        if not available_neighborhoods:
            available_neighborhoods = neighborhood_centers[:10]  # Use first 10 as fallback
        
        # Create multiple candidate locations with significant geographic spread
        for i, neighborhood in enumerate(available_neighborhoods[:6]):  # Max 6 neighborhoods per cuisine
            # Create multiple candidates per neighborhood with wider spread
            for j in range(3):  # 3 candidates per neighborhood
                # Add significant random variation around neighborhood center
                lat_offset = (np.random.random() - 0.5) * 0.025  # ±0.0125 degrees (~1.4km)
                lon_offset = (np.random.random() - 0.5) * 0.025
                
                candidate_lat = neighborhood["lat"] + lat_offset
                candidate_lon = neighborhood["lon"] + lon_offset
                
                # Ensure location is valid
                if self._is_valid_vancouver_location(candidate_lat, candidate_lon):
                    # Calculate features for this potential location
                    location_features = self._calculate_location_features(
                        candidate_lat, candidate_lon, restaurant_data, cuisine
                    )
                    
                    # Predict success score using improved model
                    success_score = self._predict_location_success(
                        location_features, cuisine, cuisine_preferences, candidate_lat, candidate_lon
                    )
                    
                    locations.append({
                        'latitude': float(candidate_lat),
                        'longitude': float(candidate_lon),
                        'success_score': float(success_score),
                        'competitor_count': int(location_features['competitor_count']),
                        'similar_cuisine_count': int(location_features['similar_cuisine_count']),
                        'confidence': float(location_features['confidence']),
                        'reason': location_features['reason'],
                        'neighborhood': neighborhood["name"]
                    })
        
        # Sort by success score and return top locations with geographic diversity
        locations.sort(key=lambda x: x['success_score'], reverse=True)
        
        # Ensure geographic diversity in recommendations
        diverse_locations = self._ensure_geographic_diversity(locations, top_n)
        return diverse_locations
    
    def _get_cuisine_preferences(self, cuisine: str) -> Dict:
        """Get cuisine-specific preferences for location scoring"""
        preferences = {
            'Italian restaurant': {'prefers_downtown': 0.7, 'competition_tolerance': 0.6},
            'Chinese restaurant': {'prefers_downtown': 0.8, 'competition_tolerance': 0.8},
            'Japanese restaurant': {'prefers_downtown': 0.6, 'competition_tolerance': 0.5},
            'Thai restaurant': {'prefers_downtown': 0.5, 'competition_tolerance': 0.6},
            'Mexican restaurant': {'prefers_downtown': 0.4, 'competition_tolerance': 0.7},
            'Indian restaurant': {'prefers_downtown': 0.6, 'competition_tolerance': 0.7},
            'Korean restaurant': {'prefers_downtown': 0.7, 'competition_tolerance': 0.5},
            'Vietnamese restaurant': {'prefers_downtown': 0.5, 'competition_tolerance': 0.8},
            'Fine dining restaurant': {'prefers_downtown': 0.9, 'competition_tolerance': 0.3},
            'Breakfast restaurant': {'prefers_downtown': 0.3, 'competition_tolerance': 0.9},
        }
        return preferences.get(cuisine, {'prefers_downtown': 0.5, 'competition_tolerance': 0.6})
    
    def _is_valid_vancouver_location(self, lat: float, lon: float) -> bool:
        """Check if location is within reasonable Vancouver bounds"""
        return (49.2 <= lat <= 49.3 and -123.25 <= lon <= -123.0)
    
    def _get_neighborhood_name(self, lat: float, lon: float) -> str:
        """Get approximate neighborhood name based on coordinates"""
        if lat > 49.27 and lon > -123.1:
            return "North Vancouver"
        elif lat > 49.26 and lon < -123.15:
            return "West End"
        elif lat > 49.25 and -123.15 <= lon <= -123.1:
            return "Downtown"
        elif lat < 49.24:
            return "South Vancouver"
        elif lon < -123.18:
            return "West Vancouver"
        else:
            return "Central Vancouver"
    
    def _ensure_geographic_diversity(self, locations: List[Dict], top_n: int) -> List[Dict]:
        """Ensure geographic diversity in top recommendations with much larger separation"""
        if len(locations) <= top_n:
            return locations
        
        diverse_locations = []
        min_distance = 0.04  # Much larger minimum distance (~4km) for true geographic diversity
        
        # Always include the top-scoring location
        if locations:
            diverse_locations.append(locations[0])
        
        # Find geographically diverse locations from remaining candidates
        for location in locations[1:]:
            if len(diverse_locations) >= top_n:
                break
                
            # Check if this location is far enough from existing recommendations
            is_diverse = True
            for existing in diverse_locations:
                distance = np.sqrt(
                    (location['latitude'] - existing['latitude'])**2 + 
                    (location['longitude'] - existing['longitude'])**2
                )
                if distance < min_distance:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_locations.append(location)
        
        # If we still need more locations, gradually relax distance requirement
        # but ensure we still have meaningful geographic spread
        fallback_distances = [0.03, 0.02, 0.015]  # Progressively smaller distances
        
        for fallback_distance in fallback_distances:
            if len(diverse_locations) >= top_n:
                break
                
            for location in locations:
                if len(diverse_locations) >= top_n:
                    break
                if location in diverse_locations:
                    continue
                    
                is_diverse = True
                for existing in diverse_locations:
                    distance = np.sqrt(
                        (location['latitude'] - existing['latitude'])**2 + 
                        (location['longitude'] - existing['longitude'])**2
                    )
                    if distance < fallback_distance:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_locations.append(location)
        
        return diverse_locations[:top_n]
    
    def _calculate_location_features(self, lat: float, lon: float, 
                                   restaurant_data: pd.DataFrame, cuisine: str) -> Dict:
        """Calculate realistic features for a potential restaurant location"""
        
        # Count nearby competitors (within 500m)
        distances = np.sqrt(
            (restaurant_data['latitude'] - lat) ** 2 + 
            (restaurant_data['longitude'] - lon) ** 2
        ) * 111000  # Convert to meters approximately
        
        nearby_mask = distances <= 500
        competitor_count = nearby_mask.sum()
        
        # Estimate similar cuisine competition more realistically
        # Use actual cuisine data if available, otherwise estimate
        if 'primary_category' in restaurant_data.columns:
            cuisine_keywords = cuisine.lower().split()
            similar_mask = restaurant_data['primary_category'].str.lower().str.contains(
                '|'.join(cuisine_keywords[:2]), na=False
            )
            similar_cuisine_count = (nearby_mask & similar_mask).sum()
        else:
            # Fallback: estimate based on cuisine popularity
            cuisine_popularity = {
                'chinese': 0.15, 'italian': 0.12, 'japanese': 0.10, 'indian': 0.08,
                'thai': 0.06, 'mexican': 0.05, 'korean': 0.04, 'vietnamese': 0.04
            }
            cuisine_key = next((k for k in cuisine_popularity.keys() if k in cuisine.lower()), 'other')
            popularity = cuisine_popularity.get(cuisine_key, 0.03)
            similar_cuisine_count = int(competitor_count * popularity)
        
        # Calculate confidence based on data density and location
        confidence = min(1.0, competitor_count / 30)  # More realistic confidence scaling
        
        # Calculate distance from downtown Vancouver
        downtown_lat, downtown_lon = 49.2827, -123.1207
        distance_from_downtown = np.sqrt(
            (lat - downtown_lat)**2 + (lon - downtown_lon)**2
        ) * 111000  # meters
        
        # Generate more nuanced reasoning
        if competitor_count < 3:
            reason = f"Emerging area for {cuisine} - first-mover advantage but unproven market"
        elif competitor_count < 10:
            reason = f"Growing market for {cuisine} - good opportunity with manageable competition"
        elif competitor_count < 25:
            reason = f"Established {cuisine} scene - proven market with moderate competition"
        elif competitor_count < 50:
            reason = f"Competitive {cuisine} market - high foot traffic but need strong differentiation"
        else:
            reason = f"Saturated {cuisine} market - very competitive, requires unique positioning"
        
        return {
            'competitor_count': int(competitor_count),
            'similar_cuisine_count': int(similar_cuisine_count),
            'distance_from_downtown': float(distance_from_downtown),
            'confidence': float(confidence),
            'reason': reason
        }
    
    def _predict_location_success(self, features: Dict, cuisine: str, 
                                preferences: Dict, lat: float, lon: float) -> float:
        """Predict success score using improved location-based model"""
        
        # Baseline success score varies by cuisine type
        base_scores = {
            'Italian restaurant': 0.65, 'Chinese restaurant': 0.70, 'Japanese restaurant': 0.68,
            'Thai restaurant': 0.64, 'Mexican restaurant': 0.62, 'Indian restaurant': 0.66,
            'Korean restaurant': 0.63, 'Vietnamese restaurant': 0.61, 'Fine dining restaurant': 0.72,
            'Breakfast restaurant': 0.58, 'Fast food': 0.55
        }
        
        cuisine_key = next((k for k in base_scores.keys() if k in cuisine), 'other')
        base_score = base_scores.get(cuisine_key, 0.60)
        
        # Competition adjustment (realistic S-curve)
        competition_count = features['competitor_count']
        if competition_count < 5:
            competition_adj = -0.05  # Too little competition might indicate poor location
        elif competition_count < 15:
            competition_adj = 0.08   # Sweet spot - proven market, manageable competition
        elif competition_count < 30:
            competition_adj = 0.03   # Competitive but viable
        elif competition_count < 50:
            competition_adj = -0.02  # Very competitive
        else:
            competition_adj = -0.08  # Oversaturated
        
        # Distance from downtown adjustment
        distance_km = features['distance_from_downtown'] / 1000
        downtown_preference = preferences['prefers_downtown']
        if distance_km < 2:  # Downtown core
            distance_adj = 0.05 * downtown_preference
        elif distance_km < 5:  # Near downtown
            distance_adj = 0.02 * downtown_preference
        elif distance_km < 10:  # Suburban
            distance_adj = -0.02 * downtown_preference
        else:  # Far suburbs
            distance_adj = -0.05 * downtown_preference
        
        # Similar cuisine density adjustment
        similar_ratio = features['similar_cuisine_count'] / max(1, competition_count)
        if similar_ratio > 0.3:  # High concentration of similar cuisine
            similar_adj = -0.03
        elif similar_ratio > 0.1:  # Some similar cuisine
            similar_adj = 0.01
        else:  # Few similar cuisine
            similar_adj = 0.02
        
        # Add some randomness for variety but keep it realistic
        random_factor = (np.random.random() - 0.5) * 0.04  # ±2% variation
        
        # Calculate final score
        final_score = base_score + competition_adj + distance_adj + similar_adj + random_factor
        
        # Ensure score is within reasonable bounds
        return max(0.3, min(0.9, final_score))
    
    def _create_recommendation_summary(self, cuisine_types: List[str], recommendations: List[Dict]):
        """Create a summary report of recommendations"""
        try:
            summary = {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_cuisines_analyzed': len(cuisine_types),
                'total_recommendations': len(recommendations),
                'cuisine_summary': {}
            }
            
            # Group recommendations by cuisine
            for cuisine in cuisine_types:
                cuisine_recs = [r for r in recommendations if r['properties']['cuisine_type'] == cuisine]
                if cuisine_recs:
                    best_rec = max(cuisine_recs, key=lambda x: x['properties']['predicted_success_score'])
                    summary['cuisine_summary'][cuisine] = {
                        'recommendations_count': len(cuisine_recs),
                        'best_location': {
                            'latitude': best_rec['geometry']['coordinates'][1],
                            'longitude': best_rec['geometry']['coordinates'][0],
                            'success_score': best_rec['properties']['predicted_success_score'],
                            'reason': best_rec['properties']['recommendation_reason']
                        }
                    }
            
            # Save summary
            summary_path = f"{self.processed_data_dir}/recommendation_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Recommendation summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error creating recommendation summary: {e}")


def main():
    """Main execution function"""
    processor = RealDataProcessor()
    processor.process_all_data()


if __name__ == "__main__":
    main()
