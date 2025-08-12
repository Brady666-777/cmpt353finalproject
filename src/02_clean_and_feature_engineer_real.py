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
                            'local_area': location.get('local_area', 'Vancouver'),
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
        """Find optimal locations for a specific cuisine type"""
        
        # Create evaluation grid
        locations = []
        
        for lat in lat_range:
            for lon in lon_range:
                # Calculate features for this potential location
                location_features = self._calculate_location_features(
                    lat, lon, restaurant_data, cuisine
                )
                
                # Predict success score (simplified model)
                success_score = self._predict_location_success(location_features, cuisine)
                
                locations.append({
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'success_score': float(success_score),
                    'competitor_count': int(location_features['competitor_count']),
                    'similar_cuisine_count': int(location_features['similar_cuisine_count']),
                    'confidence': float(location_features['confidence']),
                    'reason': location_features['reason']
                })
        
        # Sort by success score and return top locations
        locations.sort(key=lambda x: x['success_score'], reverse=True)
        return locations[:top_n]
    
    def _calculate_location_features(self, lat: float, lon: float, 
                                   restaurant_data: pd.DataFrame, cuisine: str) -> Dict:
        """Calculate features for a potential restaurant location"""
        
        # Count nearby competitors (within 500m)
        distances = np.sqrt(
            (restaurant_data['latitude'] - lat) ** 2 + 
            (restaurant_data['longitude'] - lon) ** 2
        ) * 111000  # Convert to meters approximately
        
        nearby_mask = distances <= 500
        competitor_count = nearby_mask.sum()
        
        # Estimate similar cuisine competition (simplified)
        similar_cuisine_count = max(0, competitor_count // 3)  # Assume 1/3 similar cuisine
        
        # Calculate confidence based on data density
        confidence = min(1.0, competitor_count / 20)  # Higher confidence with more data points
        
        # Generate reasoning
        if competitor_count < 5:
            reason = f"Low competition area for {cuisine} - opportunity for market entry"
        elif competitor_count > 50:
            reason = f"High-traffic area for {cuisine} - established dining district"
        else:
            reason = f"Moderate competition for {cuisine} - balanced market opportunity"
        
        return {
            'competitor_count': int(competitor_count),
            'similar_cuisine_count': int(similar_cuisine_count),
            'confidence': float(confidence),
            'reason': reason
        }
    
    def _predict_location_success(self, features: Dict, cuisine: str) -> float:
        """Predict success score for a location (simplified model)"""
        
        # Baseline success score
        base_score = 0.6
        
        # Adjust based on competition
        competition_factor = features['competitor_count']
        if competition_factor < 10:
            # Low competition - good for unique cuisines
            competition_adjustment = 0.1
        elif competition_factor > 40:
            # High competition - good for popular areas but harder to stand out
            competition_adjustment = -0.05
        else:
            # Moderate competition - balanced
            competition_adjustment = 0.05
        
        # Adjust based on cuisine popularity (simplified)
        popular_cuisines = ['Italian restaurant', 'Chinese restaurant', 'Japanese restaurant', 
                          'Indian restaurant', 'Mexican restaurant']
        
        if cuisine in popular_cuisines:
            cuisine_adjustment = 0.05  # Popular cuisines have slight advantage
        else:
            cuisine_adjustment = 0.02  # Unique cuisines in right location
        
        # Calculate final score
        success_score = base_score + competition_adjustment + cuisine_adjustment
        return max(0.1, min(1.0, success_score))  # Clamp between 0.1 and 1.0
    
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
