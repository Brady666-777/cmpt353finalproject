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
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import re
import logging
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealDataProcessor:
    """Updated data processor for real datasets"""
    
    def __init__(self):
        # Use absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = os.path.join(base_dir, 'data', 'raw')
        self.processed_data_dir = os.path.join(base_dir, 'data', 'processed')
        
        # Ensure processed data directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
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
            df = pd.read_csv(f"{self.raw_data_dir}/google-review_2025-08-06_03-55-37-484.csv")
            
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
    
    def load_combined_google_data(self) -> pd.DataFrame:
        """Load and combine both Google datasets for maximum data"""
        logger.info("Loading and combining both Google datasets...")
        
        combined_data = []
        
        # Load the reviews dataset (main one with more variation)
        try:
            df_reviews = pd.read_csv(f"{self.raw_data_dir}/google-review_2025-08-06_03-55-37-484.csv")
            logger.info(f"Loaded {len(df_reviews)} restaurants from Google reviews file")
            
            # Standardize columns
            df_reviews_clean = df_reviews.copy()
            df_reviews_clean['business_name'] = df_reviews_clean.get('title', '')
            df_reviews_clean['stars'] = pd.to_numeric(df_reviews_clean.get('totalScore', 0), errors='coerce')
            df_reviews_clean['review_count'] = pd.to_numeric(df_reviews_clean.get('reviewsCount', 0), errors='coerce')
            df_reviews_clean['primary_category'] = df_reviews_clean.get('categoryName', '')
            df_reviews_clean['address'] = df_reviews_clean.get('street', '')
            df_reviews_clean['city'] = df_reviews_clean.get('city', 'Vancouver')
            df_reviews_clean['price_level'] = 2  # Default moderate pricing
            df_reviews_clean['data_source'] = 'google_reviews'
            
            combined_data.append(df_reviews_clean)
            
        except Exception as e:
            logger.warning(f"Could not load Google reviews file: {e}")
        
        # Load the overview dataset  
        try:
            df_overview = pd.read_csv(f"{self.raw_data_dir}/good-restaurant-in-vancouver-overview.csv")
            logger.info(f"Loaded {len(df_overview)} restaurants from Google overview file")
            
            # Standardize columns
            df_overview_clean = df_overview.copy()
            df_overview_clean['business_name'] = df_overview_clean.get('name', '')
            df_overview_clean['stars'] = pd.to_numeric(df_overview_clean.get('rating', 0), errors='coerce')
            df_overview_clean['review_count'] = pd.to_numeric(df_overview_clean.get('reviews', 0), errors='coerce')
            df_overview_clean['primary_category'] = df_overview_clean.get('main_category', '')
            df_overview_clean['address'] = df_overview_clean.get('address', '')
            df_overview_clean['city'] = 'Vancouver'
            df_overview_clean['price_level'] = 2  # Default moderate pricing
            df_overview_clean['data_source'] = 'google_overview'
            
            combined_data.append(df_overview_clean)
            
        except Exception as e:
            logger.warning(f"Could not load Google overview file: {e}")
        
        if not combined_data:
            logger.error("No Google datasets could be loaded")
            return pd.DataFrame()
        
        # Combine datasets
        df_combined = pd.concat(combined_data, ignore_index=True)
        
        # Remove duplicates based on business name and address
        initial_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['business_name', 'address'], keep='first')
        final_count = len(df_combined)
        
        logger.info(f"Combined datasets: {initial_count} -> {final_count} unique restaurants (removed {initial_count - final_count} duplicates)")
        
        # Create full address
        df_combined['full_address'] = df_combined['address'] + ', ' + df_combined['city'] + ', BC'
        
        # Filter out invalid data
        df_combined = df_combined[
            (df_combined['stars'] > 0) & 
            (df_combined['stars'] <= 5) &
            (df_combined['review_count'] >= 0)
        ].copy()
        
        logger.info(f"After filtering: {len(df_combined)} valid restaurants")
        logger.info(f"Rating range: {df_combined['stars'].min():.1f} - {df_combined['stars'].max():.1f}")
        logger.info(f"Review count range: {df_combined['review_count'].min():.0f} - {df_combined['review_count'].max():.0f}")
        
        return df_combined
    
    def geocode_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Geocode addresses to get coordinates"""
        logger.info("Geocoding addresses...")
        
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        
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
        logger.info("Adding default sentiment features...")
        self._add_sentiment_features(df_clean)
        
        logger.info(f"Processed business data: {len(df_clean)} businesses")
        return df_clean
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> None:
        """Add default sentiment analysis features to the dataframe"""
        try:
            # Since InlineSentimentAnalyzer was removed, use simple default sentiment values
            # Based on business name and type to add some variation
            
            sentiment_scores = []
            sentiment_labels = []
            sentiment_confidences = []
            
            for _, row in df.iterrows():
                # Simple heuristic based on business name/type
                business_name = str(row.get('businessname', '')).lower()
                business_type = str(row.get('businesstype', '')).lower()
                
                # Basic keyword-based scoring
                positive_keywords = ['premium', 'fine', 'gourmet', 'fresh', 'best', 'quality', 'organic']
                negative_keywords = ['fast', 'quick', 'cheap', 'budget']
                
                score = 0.5  # Default neutral
                label = 'Neutral'
                confidence = 0.6
                
                # Check for positive indicators
                positive_count = sum(1 for word in positive_keywords if word in business_name or word in business_type)
                negative_count = sum(1 for word in negative_keywords if word in business_name or word in business_type)
                
                if positive_count > negative_count:
                    score = 0.65 + (positive_count * 0.05)
                    label = 'Positive'
                    confidence = 0.7
                elif negative_count > positive_count:
                    score = 0.35 - (negative_count * 0.05)
                    label = 'Negative'
                    confidence = 0.65
                
                # Ensure score bounds
                score = max(0.1, min(0.9, score))
                
                sentiment_scores.append(score)
                sentiment_labels.append(label)
                sentiment_confidences.append(confidence)
            
            # Add features to dataframe
            df['sentiment_score'] = sentiment_scores
            df['sentiment_label'] = sentiment_labels
            df['sentiment_confidence'] = sentiment_confidences
            
            logger.info(f"Added default sentiment features. Average sentiment score: {df['sentiment_score'].mean():.3f}")
                
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
        
        # Step 1: Try to load and process combined Google data first (maximum data and variation)
        google_combined = self.load_combined_google_data()
        if not google_combined.empty and len(google_combined) > 50:
            logger.info(f"Using combined Google data as primary source: {len(google_combined)} restaurants")
            
            # Geocode the Google data
            google_geocoded = self.geocode_addresses(google_combined)
            
            # Filter out restaurants without coordinates
            google_geocoded = google_geocoded.dropna(subset=['latitude', 'longitude'])
            logger.info(f"After geocoding: {len(google_geocoded)} restaurants with coordinates")
            
            if len(google_geocoded) > 0:
                # Calculate competitive features
                google_with_competition = self.calculate_competitive_features(google_geocoded)

                # Create success score
                google_final = self.create_success_score(google_with_competition)

                # Prepare model features
                X, feature_names = self.prepare_model_features(google_final)

                if not X.empty:
                    # Save processed data
                    self.save_processed_data(google_final, 'restaurants_with_features.csv')
                    self.save_processed_data(X, 'model_features.csv')

                    # Save feature names
                    pd.Series(feature_names).to_csv(
                        f"{self.processed_data_dir}/feature_names.csv", 
                        index=False, header=['feature_name']
                    )

                    logger.info("Data processing completed successfully!")
                    
                    # Print processing summary
                    print(f"\nPROCESSING SUMMARY:")
                    print(f"Total restaurants processed: {len(google_final)}")
                    print(f"Features created: {len(feature_names)}")
                    print(f"Feature names: {feature_names}")
                    print(f"Success score range: {google_final['success_score'].min():.3f} - {google_final['success_score'].max():.3f}")
                    print(f"Average success score: {google_final['success_score'].mean():.3f}")
                    return

        # Fallback: Try individual Google reviews data
        google_reviews = self.load_google_reviews_data()
        if not google_reviews.empty and len(google_reviews) > 50:
            logger.info(f"Using Google reviews data as primary source: {len(google_reviews)} restaurants")
            
            # Geocode the Google reviews data
            google_geocoded = self.geocode_addresses(google_reviews)
            
            # Filter out restaurants without coordinates
            google_geocoded = google_geocoded.dropna(subset=['latitude', 'longitude'])
            logger.info(f"After geocoding: {len(google_geocoded)} restaurants with coordinates")
            
            if len(google_geocoded) > 0:
                # Calculate competitive features
                google_with_competition = self.calculate_competitive_features(google_geocoded)

                # Create success score
                google_final = self.create_success_score(google_with_competition)

                # Prepare model features
                X, feature_names = self.prepare_model_features(google_final)

                if not X.empty:
                    # Save processed data
                    self.save_processed_data(google_final, 'restaurants_with_features.csv')
                    self.save_processed_data(X, 'model_features.csv')

                    # Save feature names
                    pd.Series(feature_names).to_csv(
                        f"{self.processed_data_dir}/feature_names.csv", 
                        index=False, header=['feature_name']
                    )

                    logger.info("Data processing completed successfully!")
                    
                    # Print processing summary
                    print(f"\nPROCESSING SUMMARY:")
                    print(f"Total restaurants processed: {len(google_final)}")
                    print(f"Features created: {len(feature_names)}")
                    print(f"Feature names: {feature_names}")
                    print(f"Success score range: {google_final['success_score'].min():.3f} - {google_final['success_score'].max():.3f}")
                    print(f"Average success score: {google_final['success_score'].mean():.3f}")
                    return

        # Fallback: Load and process business licenses if Google data not available
        logger.info("Falling back to business license data...")
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

def main():
    """Main execution function"""
    processor = RealDataProcessor()
    processor.process_all_data()


if __name__ == "__main__":
    main()
