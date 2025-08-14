"""
Data Collection Script for VancouverPy Restaurant Success Prediction

This script handles the organization and validation of existing data files:
- Business Licenses data (existing GeoJSON file)
- Census data (existing CSV file) 
- Restaurant data (existing CSV files)

Author: VancouverPy Project Team
Date: August 2025
"""

import os
import pandas as pd
import geopandas as gpd
import shutil
from pathlib import Path
import logging
import json

# Set up logging
project_root = Path(__file__).parent.parent
log_file = project_root / 'data' / 'data_collection.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Main class for organizing and validating existing data files"""
    
    def __init__(self):
        # Get the project root directory (parent of src)
        self.project_root = Path(__file__).parent.parent
        self.raw_data_dir = self.project_root / 'data' / 'raw'
        self.processed_data_dir = self.project_root / 'data' / 'processed'
        
        # Ensure data directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected raw data files
        self.expected_files = {
            'business_licenses': 'business-licences.geojson',
            'census_data': 'CensusProfile2021-ProfilRecensement2021-20250811051126.csv',
            'restaurant_overview': 'good-restaurant-in-vancouver-overview.csv',
            'restaurant_reviews': 'google-review_2025-08-06_03-55-37-484.csv'
        }

    def validate_raw_data(self) -> dict:
        """
        Validate that all expected raw data files exist and are readable
        """
        logger.info("Validating raw data files...")
        
        validation_results = {}
        
        for data_type, filename in self.expected_files.items():
            file_path = self.raw_data_dir / filename
            
            if file_path.exists():
                try:
                    if filename.endswith('.geojson'):
                        # Validate GeoJSON file
                        gdf = gpd.read_file(file_path)
                        validation_results[data_type] = {
                            'exists': True,
                            'readable': True,
                            'rows': len(gdf),
                            'columns': list(gdf.columns),
                            'file_size_mb': file_path.stat().st_size / (1024*1024)
                        }
                        logger.info(f"[OK] {data_type}: {len(gdf)} records loaded from {filename}")
                        
                    elif filename.endswith('.csv'):
                        # Validate CSV file with different encodings
                        df = None
                        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if df is not None:
                            validation_results[data_type] = {
                                'exists': True,
                                'readable': True,
                                'rows': len(df),
                                'columns': list(df.columns),
                                'file_size_mb': file_path.stat().st_size / (1024*1024),
                                'encoding_used': encoding
                            }
                            logger.info(f"[OK] {data_type}: {len(df)} records loaded from {filename} (encoding: {encoding})")
                        else:
                            validation_results[data_type] = {
                                'exists': True,
                                'readable': False,
                                'error': 'Could not decode with any common encoding'
                            }
                            logger.error(f"[ERROR] {data_type}: Could not decode file with any common encoding")
                        
                except Exception as e:
                    validation_results[data_type] = {
                        'exists': True,
                        'readable': False,
                        'error': str(e)
                    }
                    logger.error(f"[ERROR] {data_type}: File exists but cannot be read - {e}")
                    
            else:
                validation_results[data_type] = {
                    'exists': False,
                    'readable': False,
                    'error': 'File not found'
                }
                logger.error(f"[ERROR] {data_type}: File {filename} not found")
        
        return validation_results

    def load_business_licenses(self) -> gpd.GeoDataFrame:
        """
        Load and analyze business license data
        Focus on food service establishments
        """
        logger.info("Loading business license data...")
        
        file_path = self.raw_data_dir / self.expected_files['business_licenses']
        
        try:
            gdf = gpd.read_file(file_path)
            
            # Filter for food service establishments
            food_keywords = ['restaurant', 'cafe', 'bakery', 'food', 'bar', 'bistro', 'grill', 'pizza']
            
            # Check in businesstype and businesstradename columns
            if 'businesstype' in gdf.columns:
                food_mask = gdf['businesstype'].str.contains('|'.join(food_keywords), case=False, na=False)
            else:
                food_mask = pd.Series([False] * len(gdf))
                
            if 'businesstradename' in gdf.columns:
                food_mask |= gdf['businesstradename'].str.contains('|'.join(food_keywords), case=False, na=False)
            
            if 'businessname' in gdf.columns:
                food_mask |= gdf['businessname'].str.contains('|'.join(food_keywords), case=False, na=False)
            
            food_establishments = gdf[food_mask].copy()
            
            # Save filtered data
            output_path = self.processed_data_dir / 'business_licenses_food.geojson'
            food_establishments.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"Filtered {len(food_establishments)} food establishments from {len(gdf)} total licenses")
            return food_establishments
            
        except Exception as e:
            logger.error(f"Error loading business licenses: {e}")
            return gpd.GeoDataFrame()

    def load_census_data(self) -> pd.DataFrame:
        """
        Load and process census data
        """
        logger.info("Loading census data...")
        
        file_path = self.raw_data_dir / self.expected_files['census_data']
        
        try:
            # The census file has a specific format, let's examine its structure
            # Try different encodings
            df = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Successfully loaded census data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is not None:
                # Save basic info about the census data
                output_path = self.processed_data_dir / 'census_data_raw.csv'
                df.to_csv(output_path, index=False, encoding='utf-8')
                
                logger.info(f"Loaded census data with {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                logger.error("Could not load census data with any encoding")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading census data: {e}")
            return pd.DataFrame()

    def load_restaurant_data(self) -> tuple:
        """
        Load restaurant overview and review data
        """
        logger.info("Loading restaurant data...")
        
        overview_path = self.raw_data_dir / self.expected_files['restaurant_overview']
        reviews_path = self.raw_data_dir / self.expected_files['restaurant_reviews']
        
        overview_df = pd.DataFrame()
        reviews_df = pd.DataFrame()
        
        try:
            # Load restaurant overview data
            overview_df = pd.read_csv(overview_path)
            overview_output = self.processed_data_dir / 'google_restaurants_overview.csv'
            overview_df.to_csv(overview_output, index=False)
            logger.info(f"Loaded restaurant overview data: {len(overview_df)} restaurants")
            
        except Exception as e:
            logger.error(f"Error loading restaurant overview data: {e}")
        
        try:
            # Load restaurant reviews data
            reviews_df = pd.read_csv(reviews_path)
            reviews_output = self.processed_data_dir / 'google_restaurants_reviews.csv'
            reviews_df.to_csv(reviews_output, index=False)
            logger.info(f"Loaded restaurant reviews data: {len(reviews_df)} restaurants")
            
        except Exception as e:
            logger.error(f"Error loading restaurant reviews data: {e}")
        
        return overview_df, reviews_df

    def generate_data_summary(self, validation_results: dict) -> dict:
        """
        Generate a comprehensive summary of the data collection process
        """
        logger.info("Generating data summary...")
        
        summary = {
            'data_collection_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_results': validation_results,
            'files_processed': {
                'business_licenses_food.geojson': 'Filtered food establishments from business licenses',
                'census_data_raw.csv': 'Raw census data for Vancouver',
                'google_restaurants_overview.csv': 'Restaurant overview data from Google',
                'google_restaurants_reviews.csv': 'Restaurant reviews data from Google'
            },
            'next_steps': [
                'Run data cleaning and feature engineering (02_clean_and_feature_engineer.py)',
                'Perform model training (03_model_training.py)',
                'Generate visualizations and reports'
            ]
        }
        
        # Save summary
        summary_path = self.processed_data_dir / 'data_collection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

    def organize_all_data(self):
        """
        Main method to organize and validate all data files
        """
        logger.info("Starting data organization and validation...")
        
        # Validate raw data
        validation_results = self.validate_raw_data()
        
        # Process each data source
        business_licenses = self.load_business_licenses()
        census_data = self.load_census_data()
        overview_df, reviews_df = self.load_restaurant_data()
        
        # Generate summary
        summary = self.generate_data_summary(validation_results)
        
        logger.info("Data organization completed!")
        logger.info(f"Summary saved to: {self.processed_data_dir / 'data_collection_summary.json'}")
        
        return {
            'validation_results': validation_results,
            'business_licenses': business_licenses,
            'census_data': census_data,
            'overview_data': overview_df,
            'reviews_data': reviews_df,
            'summary': summary
        }


def main():
    """
    Main execution function
    """
    collector = DataCollector()
    results = collector.organize_all_data()
    
    # Print summary
    print("\n" + "="*60)
    print("DATA ORGANIZATION SUMMARY")
    print("="*60)
    
    validation_results = results['validation_results']
    for data_type, result in validation_results.items():
        status = "[OK]" if result.get('exists') and result.get('readable') else "[ERROR]"
        print(f"{status} {data_type.replace('_', ' ').title()}")
        if result.get('readable'):
            print(f"   - Rows: {result.get('rows', 'N/A')}")
            print(f"   - File size: {result.get('file_size_mb', 0):.2f} MB")
            if result.get('encoding_used'):
                print(f"   - Encoding: {result.get('encoding_used')}")
    
    print("\nProcessed files saved to: data/processed/")
    print("Summary file: data/processed/data_collection_summary.json")


if __name__ == "__main__":
    main()
