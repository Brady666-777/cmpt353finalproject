"""
Data Cleaning and Feature Engineering Script for VancouverPy Restaurant Success Prediction

This script handles:
1. Data cleaning and standardization
2. Address geocoding and spatial joins
3. Feature engineering for machine learning
4. Target variable creation (Success Score)

Author: VancouverPy Project Team
Date: August 2025
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
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

class DataProcessor:
    """Main class for data cleaning and feature engineering"""
    
    def __init__(self):
        self.raw_data_dir = '../data/raw'
        self.processed_data_dir = '../data/processed'
        
        # Ensure processed data directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="vancouvpy_restaurant_predictor")
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all raw datasets
        """
        logger.info("Loading raw datasets...")
        
        datasets = {}
        
        try:
            datasets['business_licenses'] = pd.read_csv(f"{self.raw_data_dir}/business_licenses.csv")
            datasets['yelp_restaurants'] = pd.read_csv(f"{self.raw_data_dir}/yelp_restaurants.csv")
            datasets['traffic_counts'] = pd.read_csv(f"{self.raw_data_dir}/traffic_counts.csv")
            datasets['census_data'] = pd.read_csv(f"{self.raw_data_dir}/census_data.csv")
            datasets['transit_stops'] = pd.read_csv(f"{self.raw_data_dir}/transit_stops.csv")
            
            # Load geospatial data
            datasets['local_areas'] = gpd.read_file(f"{self.raw_data_dir}/local_areas.geojson")
            
            logger.info("Raw datasets loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Raw data file not found: {e}")
            logger.info("Please run 01_get_data.py first to collect the data")
            
        return datasets
    
    def clean_business_licenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize business license data
        """
        logger.info("Cleaning business license data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        
        # Filter for active food establishments
        food_keywords = ['restaurant', 'cafe', 'bakery', 'food', 'coffee', 'bar', 'pub']
        food_filter = df_clean['businesstype'].str.lower().str.contains('|'.join(food_keywords), na=False)
        active_filter = df_clean['status'].str.lower() == 'issued'
        
        df_clean = df_clean[food_filter & active_filter].copy()
        
        # Clean address data
        if 'businessaddress' in df_clean.columns:
            df_clean['address_clean'] = df_clean['businessaddress'].str.strip().str.title()
            # Add Vancouver, BC for geocoding
            df_clean['full_address'] = df_clean['address_clean'] + ', Vancouver, BC, Canada'
        
        # Clean business names
        if 'businessname' in df_clean.columns:
            df_clean['business_name_clean'] = df_clean['businessname'].str.strip().str.title()
        
        logger.info(f"Cleaned business licenses: {len(df_clean)} records remaining")
        return df_clean
    
    def clean_yelp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize Yelp restaurant data
        """
        logger.info("Cleaning Yelp restaurant data...")
        
        df_clean = df.copy()
        
        # Extract coordinates from location data if nested
        if 'coordinates' in df_clean.columns:
            # Handle nested coordinate data
            if isinstance(df_clean['coordinates'].iloc[0], dict):
                df_clean['latitude'] = df_clean['coordinates'].apply(lambda x: x.get('latitude') if isinstance(x, dict) else None)
                df_clean['longitude'] = df_clean['coordinates'].apply(lambda x: x.get('longitude') if isinstance(x, dict) else None)
        
        # Extract price level
        if 'price' in df_clean.columns:
            df_clean['price_level'] = df_clean['price'].str.len()
            df_clean['price_level'] = df_clean['price_level'].fillna(2)  # Default to moderate pricing
        
        # Clean categories
        if 'categories' in df_clean.columns:
            df_clean['primary_category'] = df_clean['categories'].apply(
                lambda x: x[0]['title'] if isinstance(x, list) and len(x) > 0 else 'Restaurant'
            )
        
        # Filter for Vancouver restaurants only
        if 'location' in df_clean.columns:
            vancouver_filter = df_clean['location'].astype(str).str.contains('Vancouver', na=False)
            df_clean = df_clean[vancouver_filter].copy()
        
        logger.info(f"Cleaned Yelp data: {len(df_clean)} records remaining")
        return df_clean
    
    def geocode_addresses(self, df: pd.DataFrame, address_col: str) -> pd.DataFrame:
        """
        Geocode addresses to get latitude and longitude
        """
        logger.info("Geocoding addresses...")
        
        df_geo = df.copy()
        
        # Initialize coordinate columns
        df_geo['latitude'] = None
        df_geo['longitude'] = None
        
        # Geocode addresses (with rate limiting)
        for idx, address in enumerate(df_geo[address_col].dropna()):
            if idx % 100 == 0:
                logger.info(f"Geocoded {idx} addresses...")
            
            try:
                location = self.geolocator.geocode(address, timeout=10)
                if location:
                    df_geo.loc[df_geo[address_col] == address, 'latitude'] = location.latitude
                    df_geo.loc[df_geo[address_col] == address, 'longitude'] = location.longitude
                    
            except Exception as e:
                logger.warning(f"Geocoding failed for {address}: {e}")
                continue
        
        # Remove records without coordinates
        initial_count = len(df_geo)
        df_geo = df_geo.dropna(subset=['latitude', 'longitude'])
        logger.info(f"Geocoding complete: {len(df_geo)}/{initial_count} records with coordinates")
        
        return df_geo
    
    def calculate_competitive_density(self, df: pd.DataFrame, radius_km: float = 0.5) -> pd.DataFrame:
        """
        Calculate competitive density around each restaurant
        """
        logger.info(f"Calculating competitive density within {radius_km}km...")
        
        df_comp = df.copy()
        df_comp['competitor_count'] = 0
        df_comp['similar_cuisine_count'] = 0
        
        # Create GeoDataFrame for spatial operations
        gdf = gpd.GeoDataFrame(
            df_comp, 
            geometry=gpd.points_from_xy(df_comp.longitude, df_comp.latitude),
            crs='EPSG:4326'
        )
        
        # Convert to projected CRS for distance calculations
        gdf_proj = gdf.to_crs('EPSG:3157')  # BC Albers projection
        
        # Calculate buffers
        buffers = gdf_proj.geometry.buffer(radius_km * 1000)  # Convert km to meters
        
        # Count competitors within buffer
        for idx, (point, buffer_geom) in enumerate(zip(gdf_proj.geometry, buffers)):
            if idx % 100 == 0:
                logger.info(f"Processed {idx} restaurants for competitive analysis...")
            
            # Find restaurants within buffer (excluding self)
            within_buffer = gdf_proj[gdf_proj.geometry.within(buffer_geom) & (gdf_proj.index != idx)]
            
            df_comp.loc[idx, 'competitor_count'] = len(within_buffer)
            
            # Count similar cuisine types
            if 'primary_category' in df_comp.columns:
                current_category = df_comp.loc[idx, 'primary_category']
                similar_cuisine = within_buffer[within_buffer['primary_category'] == current_category]
                df_comp.loc[idx, 'similar_cuisine_count'] = len(similar_cuisine)
        
        logger.info("Competitive density calculation complete")
        return df_comp
    
    def calculate_transit_accessibility(self, df: pd.DataFrame, transit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transit accessibility metrics
        """
        logger.info("Calculating transit accessibility...")
        
        df_transit = df.copy()
        df_transit['nearest_station_distance'] = None
        df_transit['bus_stops_500m'] = 0
        
        # Calculate distance to nearest transit station
        for idx, row in df_transit.iterrows():
            restaurant_coords = (row['latitude'], row['longitude'])
            
            # Calculate distances to all transit stops
            distances = []
            for _, stop in transit_df.iterrows():
                stop_coords = (stop['stop_lat'], stop['stop_lon'])
                distance = geodesic(restaurant_coords, stop_coords).kilometers
                distances.append(distance)
            
            # Store nearest station distance
            df_transit.loc[idx, 'nearest_station_distance'] = min(distances) if distances else None
            
            # Count bus stops within 500m
            nearby_stops = sum(1 for d in distances if d <= 0.5)
            df_transit.loc[idx, 'bus_stops_500m'] = nearby_stops
        
        logger.info("Transit accessibility calculation complete")
        return df_transit
    
    def create_neighborhood_features(self, df: pd.DataFrame, census_df: pd.DataFrame, 
                                   local_areas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Create neighborhood-level features
        """
        logger.info("Creating neighborhood features...")
        
        df_neighborhood = df.copy()
        
        # Create GeoDataFrame for spatial join
        gdf_restaurants = gpd.GeoDataFrame(
            df_neighborhood,
            geometry=gpd.points_from_xy(df_neighborhood.longitude, df_neighborhood.latitude),
            crs='EPSG:4326'
        )
        
        # Spatial join with local areas
        gdf_with_areas = gpd.sjoin(gdf_restaurants, local_areas_gdf, how='left', predicate='within')
        
        # Merge with census data
        if 'local_area' in census_df.columns:
            gdf_final = gdf_with_areas.merge(census_df, on='local_area', how='left')
        else:
            gdf_final = gdf_with_areas
        
        # Create affordability mismatch feature
        if 'price_level' in df_neighborhood.columns and 'median_income' in census_df.columns:
            # Normalize income to price level scale (1-4)
            income_normalized = (gdf_final['median_income'] - gdf_final['median_income'].min()) / \
                               (gdf_final['median_income'].max() - gdf_final['median_income'].min()) * 3 + 1
            
            gdf_final['affordability_mismatch'] = abs(gdf_final['price_level'] - income_normalized)
        
        logger.info("Neighborhood features created")
        return pd.DataFrame(gdf_final.drop('geometry', axis=1))
    
    def create_success_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target variable: Success Score
        """
        logger.info("Creating Success Score target variable...")
        
        df_target = df.copy()
        
        # Components of success score
        # 1. Rating (normalized to 0-1)
        if 'rating' in df_target.columns:
            rating_normalized = (df_target['rating'] - 1) / 4  # Yelp ratings are 1-5
        else:
            rating_normalized = 0.5  # Default neutral rating
        
        # 2. Review count (log-normalized)
        if 'review_count' in df_target.columns:
            review_log = np.log1p(df_target['review_count'])
            review_normalized = (review_log - review_log.min()) / (review_log.max() - review_log.min())
        else:
            review_normalized = 0.5  # Default
        
        # 3. Longevity proxy (if available)
        longevity_score = 0.5  # Placeholder - would need business age data
        
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
        """
        Prepare final feature set for modeling
        """
        logger.info("Preparing model features...")
        
        # Define feature columns
        feature_columns = [
            'latitude', 'longitude', 'price_level',
            'competitor_count', 'similar_cuisine_count',
            'nearest_station_distance', 'bus_stops_500m',
            'population', 'median_income', 'affordability_mismatch'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_columns if col in df.columns]
        
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
        return X, available_features
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to CSV
        """
        filepath = f"{self.processed_data_dir}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
    
    def process_all_data(self):
        """
        Main processing pipeline
        """
        logger.info("Starting comprehensive data processing...")
        
        # Load raw data
        datasets = self.load_raw_data()
        
        if not datasets:
            logger.error("No datasets loaded. Exiting.")
            return
        
        # Process business licenses
        if 'business_licenses' in datasets:
            business_clean = self.clean_business_licenses(datasets['business_licenses'])
            
            # Geocode if no coordinates
            if 'latitude' not in business_clean.columns:
                business_clean = self.geocode_addresses(business_clean, 'full_address')
            
            self.save_processed_data(business_clean, 'business_licenses_clean.csv')
        
        # Process Yelp data
        if 'yelp_restaurants' in datasets:
            yelp_clean = self.clean_yelp_data(datasets['yelp_restaurants'])
            
            # Calculate competitive features
            yelp_with_competition = self.calculate_competitive_density(yelp_clean)
            
            # Calculate transit accessibility
            if 'transit_stops' in datasets:
                yelp_with_transit = self.calculate_transit_accessibility(
                    yelp_with_competition, datasets['transit_stops']
                )
            else:
                yelp_with_transit = yelp_with_competition
            
            # Add neighborhood features
            if 'census_data' in datasets and 'local_areas' in datasets:
                yelp_with_neighborhoods = self.create_neighborhood_features(
                    yelp_with_transit, datasets['census_data'], datasets['local_areas']
                )
            else:
                yelp_with_neighborhoods = yelp_with_transit
            
            # Create success score
            yelp_final = self.create_success_score(yelp_with_neighborhoods)
            
            # Prepare model features
            X, feature_names = self.prepare_model_features(yelp_final)
            
            # Save processed data
            self.save_processed_data(yelp_final, 'restaurants_with_features.csv')
            self.save_processed_data(X, 'model_features.csv')
            
            # Save feature names
            pd.Series(feature_names).to_csv(f"{self.processed_data_dir}/feature_names.csv", 
                                          index=False, header=['feature'])
        
        logger.info("Data processing completed!")


def main():
    """
    Main execution function
    """
    processor = DataProcessor()
    processor.process_all_data()


if __name__ == "__main__":
    main()
