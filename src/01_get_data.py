"""
Generalized Data Collection Script for Data Science Projects

This script provides a flexible framework for organizing and validating data files:
- Support for multiple file types (CSV, GeoJSON, JSON)
- Configurable data processing pipelines
- Automatic encoding detection for CSV files
- Flexible filtering and transformation capabilities
- Easy customization for different projects

Example Usage:
    # Default configuration
    collector = DataCollector()
    results = collector.organize_all_data()
    
    # Custom configuration
    custom_config = {
        'my_data': {
            'filename': 'data.csv',
            'type': 'csv',
            'output_name': 'processed_data.csv'
        }
    }
    collector = DataCollector(data_config=custom_config)
    
    # Dynamic data source addition
    collector.add_data_source('new_data', 'file.json', 'json')

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
    """Generalized class for organizing and validating data files"""
    
    def __init__(self, data_config=None, project_root=None):
        # Get the project root directory (parent of src)
        self.project_root = project_root or Path(__file__).parent.parent
        self.raw_data_dir = self.project_root / 'data' / 'raw'
        self.processed_data_dir = self.project_root / 'data' / 'processed'
        
        # Ensure data directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided data config or default to Vancouver restaurant data
        self.data_config = data_config or self._get_default_vancouver_config()
    
    def _get_default_vancouver_config(self):
        """Default configuration for Vancouver restaurant data"""
        return {
            'business_licenses': {
                'filename': 'business-licences.geojson',
                'type': 'geojson',
                'filter_keywords': ['restaurant', 'cafe', 'bakery', 'food', 'bar', 'bistro', 'grill', 'pizza'],
                'filter_columns': ['businesstype', 'businesstradename', 'businessname'],
                'output_name': 'business_licenses_food.geojson'
            },
            'census_data': {
                'filename': 'CensusProfile2021-ProfilRecensement2021-20250811051126.csv',
                'type': 'csv',
                'output_name': 'census_data_raw.csv'
            },
            'restaurant_overview': {
                'filename': 'good-restaurant-in-vancouver-overview.csv',
                'type': 'csv',
                'output_name': 'google_restaurants_overview.csv'
            },
            'restaurant_reviews': {
                'filename': 'google-review_2025-08-06_03-55-37-484.csv',
                'type': 'csv',
                'output_name': 'google_restaurants_reviews.csv'
            }
        }

    def validate_raw_data(self) -> dict:
        """
        Validate that all expected raw data files exist and are readable
        """
        logger.info("Validating raw data files...")
        
        validation_results = {}
        
        for data_type, config in self.data_config.items():
            filename = config['filename']
            file_type = config['type']
            file_path = self.raw_data_dir / filename
            
            if file_path.exists():
                try:
                    if file_type == 'geojson':
                        # Validate GeoJSON file
                        gdf = gpd.read_file(file_path)
                        validation_results[data_type] = {
                            'exists': True,
                            'readable': True,
                            'rows': len(gdf),
                            'columns': list(gdf.columns),
                            'file_size_mb': file_path.stat().st_size / (1024*1024),
                            'type': file_type
                        }
                        logger.info(f"[OK] {data_type}: {len(gdf)} records loaded from {filename}")
                        
                    elif file_type == 'csv':
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
                                'encoding_used': encoding,
                                'type': file_type
                            }
                            logger.info(f"[OK] {data_type}: {len(df)} records loaded from {filename} (encoding: {encoding})")
                        else:
                            validation_results[data_type] = {
                                'exists': True,
                                'readable': False,
                                'error': 'Could not decode with any common encoding',
                                'type': file_type
                            }
                            logger.error(f"[ERROR] {data_type}: Could not decode file with any common encoding")
                    
                    elif file_type == 'json':
                        # Validate JSON file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        validation_results[data_type] = {
                            'exists': True,
                            'readable': True,
                            'records': len(data) if isinstance(data, list) else 1,
                            'file_size_mb': file_path.stat().st_size / (1024*1024),
                            'type': file_type
                        }
                        logger.info(f"[OK] {data_type}: JSON file loaded from {filename}")
                        
                except Exception as e:
                    validation_results[data_type] = {
                        'exists': True,
                        'readable': False,
                        'error': str(e),
                        'type': file_type
                    }
                    logger.error(f"[ERROR] {data_type}: File exists but cannot be read - {e}")
                    
            else:
                validation_results[data_type] = {
                    'exists': False,
                    'readable': False,
                    'error': 'File not found',
                    'type': file_type
                }
                logger.error(f"[ERROR] {data_type}: File {filename} not found")
        
        return validation_results

    def load_and_process_data(self, data_type: str = None) -> dict:
        """
        Load and process data based on configuration
        If data_type is specified, only process that data type
        """
        logger.info(f"Loading and processing data{'for ' + data_type if data_type else ''}...")
        
        results = {}
        data_configs = {data_type: self.data_config[data_type]} if data_type else self.data_config
        
        for dtype, config in data_configs.items():
            try:
                if config['type'] == 'geojson':
                    results[dtype] = self._process_geojson(dtype, config)
                elif config['type'] == 'csv':
                    results[dtype] = self._process_csv(dtype, config)
                elif config['type'] == 'json':
                    results[dtype] = self._process_json(dtype, config)
                else:
                    logger.warning(f"Unknown data type '{config['type']}' for {dtype}")
                    results[dtype] = None
            except Exception as e:
                logger.error(f"Error processing {dtype}: {e}")
                results[dtype] = None
        
        return results

    def _process_geojson(self, data_type: str, config: dict) -> gpd.GeoDataFrame:
        """Process GeoJSON files with optional filtering"""
        logger.info(f"Processing GeoJSON data: {data_type}")
        
        file_path = self.raw_data_dir / config['filename']
        gdf = gpd.read_file(file_path)
        
        # Apply filtering if specified
        if 'filter_keywords' in config and 'filter_columns' in config:
            filter_keywords = config['filter_keywords']
            filter_columns = config['filter_columns']
            
            # Create filter mask
            mask = pd.Series([False] * len(gdf))
            for col in filter_columns:
                if col in gdf.columns:
                    mask |= gdf[col].str.contains('|'.join(filter_keywords), case=False, na=False)
            
            gdf = gdf[mask].copy()
            logger.info(f"Filtered to {len(gdf)} records using keywords: {filter_keywords}")
        
        # Save processed data
        output_path = self.processed_data_dir / config['output_name']
        gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Saved processed data to: {config['output_name']}")
        
        return gdf

    def _process_csv(self, data_type: str, config: dict) -> pd.DataFrame:
        """Process CSV files with automatic encoding detection"""
        logger.info(f"Processing CSV data: {data_type}")
        
        file_path = self.raw_data_dir / config['filename']
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded {data_type} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not decode {config['filename']} with any common encoding")
        
        # Apply any specified transformations
        if 'transformations' in config:
            for transform in config['transformations']:
                if transform['type'] == 'filter_columns':
                    df = df[transform['columns']]
                elif transform['type'] == 'rename_columns':
                    df = df.rename(columns=transform['mapping'])
                elif transform['type'] == 'filter_rows':
                    query = transform['query']
                    df = df.query(query)
        
        # Save processed data
        output_path = self.processed_data_dir / config['output_name']
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved processed data to: {config['output_name']}")
        
        return df

    def _process_json(self, data_type: str, config: dict) -> dict:
        """Process JSON files"""
        logger.info(f"Processing JSON data: {data_type}")
        
        file_path = self.raw_data_dir / config['filename']
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Save processed data
        output_path = self.processed_data_dir / config['output_name']
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to: {config['output_name']}")
        
        return data

    def list_raw_files(self) -> dict:
        """List all files in the raw data directory"""
        files = {}
        for file_path in self.raw_data_dir.glob('*'):
            if file_path.is_file():
                files[file_path.name] = {
                    'size_mb': file_path.stat().st_size / (1024*1024),
                    'extension': file_path.suffix,
                    'path': str(file_path)
                }
        return files

    def suggest_config_from_files(self) -> dict:
        """Generate suggested configuration based on files in raw directory"""
        files = self.list_raw_files()
        suggested_config = {}
        
        for filename, info in files.items():
            name = filename.split('.')[0].lower().replace('-', '_').replace(' ', '_')
            
            if info['extension'] == '.geojson':
                file_type = 'geojson'
                output_name = f"{name}.geojson"
            elif info['extension'] == '.csv':
                file_type = 'csv'
                output_name = f"{name}.csv"
            elif info['extension'] == '.json':
                file_type = 'json'
                output_name = f"{name}.json"
            else:
                continue  # Skip unsupported file types
            
            suggested_config[name] = {
                'filename': filename,
                'type': file_type,
                'output_name': output_name
            }
        
        return suggested_config

    def export_config(self, filepath: str = None):
        """Export current configuration to a JSON file"""
        filepath = filepath or str(self.processed_data_dir / 'data_config.json')
        with open(filepath, 'w') as f:
            json.dump(self.data_config, f, indent=2)
        logger.info(f"Configuration exported to: {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from a JSON file"""
        with open(filepath, 'r') as f:
            self.data_config = json.load(f)
        logger.info(f"Configuration loaded from: {filepath}")

    def update_data_config(self, new_config: dict):
        """Update the data configuration"""
        self.data_config.update(new_config)
        logger.info(f"Updated data configuration with {len(new_config)} entries")

    def add_data_source(self, name: str, filename: str, file_type: str, output_name: str = None, **kwargs):
        """Add a new data source to the configuration"""
        config = {
            'filename': filename,
            'type': file_type,
            'output_name': output_name or filename,
            **kwargs
        }
        self.data_config[name] = config
        logger.info(f"Added data source '{name}' of type '{file_type}'")

    def generate_data_summary(self, validation_results: dict, processed_data: dict) -> dict:
        """
        Generate a comprehensive summary of the data collection process
        """
        logger.info("Generating data summary...")
        
        files_processed = {}
        for data_type, config in self.data_config.items():
            if processed_data.get(data_type) is not None:
                files_processed[config['output_name']] = f"Processed {data_type} data"
        
        summary = {
            'data_collection_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'validation_results': validation_results,
            'files_processed': files_processed,
            'data_config': self.data_config,
            'next_steps': [
                'Run data cleaning and feature engineering (02_clean_and_feature_engineer.py)',
                'Perform model training (03_model_training.py)',
                'Generate visualizations and reports'
            ]
        }
        
        # Save summary
        summary_path = self.processed_data_dir / 'data_collection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary

    def organize_all_data(self):
        """
        Main method to organize and validate all data files
        """
        logger.info("Starting data organization and validation...")
        
        # Validate raw data
        validation_results = self.validate_raw_data()
        
        # Process all data sources
        processed_data = self.load_and_process_data()
        
        # Generate summary
        summary = self.generate_data_summary(validation_results, processed_data)
        
        logger.info("Data organization completed!")
        logger.info(f"Summary saved to: {self.processed_data_dir / 'data_collection_summary.json'}")
        
        return {
            'validation_results': validation_results,
            'processed_data': processed_data,
            'summary': summary
        }


def main():
    """
    Main execution function demonstrating flexible data collection
    """
    # Example 1: Default Vancouver restaurant data
    print("\n" + "="*60)
    print("EXAMPLE 1: Default Vancouver Restaurant Data")
    print("="*60)
    
    collector = DataCollector()
    results = collector.organize_all_data()
    
    # Print summary
    validation_results = results['validation_results']
    for data_type, result in validation_results.items():
        status = "[OK]" if result.get('exists') and result.get('readable') else "[ERROR]"
        print(f"{status} {data_type.replace('_', ' ').title()}")
        if result.get('readable'):
            rows_or_records = result.get('rows') or result.get('records', 'N/A')
            print(f"   - Records: {rows_or_records}")
            print(f"   - File size: {result.get('file_size_mb', 0):.2f} MB")
            if result.get('encoding_used'):
                print(f"   - Encoding: {result.get('encoding_used')}")
    
    # Example 2: Custom configuration for different data types
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Data Configuration")
    print("="*60)
    
    # Create a custom configuration
    custom_config = {
        'custom_geojson': {
            'filename': 'my_custom_data.geojson',
            'type': 'geojson',
            'output_name': 'processed_custom_geo.geojson'
        },
        'custom_csv': {
            'filename': 'my_data.csv',
            'type': 'csv',
            'output_name': 'processed_data.csv',
            'transformations': [
                {
                    'type': 'filter_columns',
                    'columns': ['id', 'name', 'value']
                },
                {
                    'type': 'rename_columns',
                    'mapping': {'old_name': 'new_name'}
                }
            ]
        }
    }
    
    # Initialize with custom config
    custom_collector = DataCollector(data_config=custom_config)
    
    # Or add data sources dynamically
    custom_collector.add_data_source(
        name='api_data',
        filename='api_response.json',
        file_type='json',
        output_name='processed_api_data.json'
    )
    
    print(f"Custom collector configured with {len(custom_collector.data_config)} data sources")
    
    # Example 3: Adding filter configuration
    print("\n" + "="*60)
    print("EXAMPLE 3: Filtered Data Processing")
    print("="*60)
    
    # Add a filtered business license configuration
    filtered_collector = DataCollector()
    filtered_collector.add_data_source(
        name='tech_businesses',
        filename='business-licences.geojson',
        file_type='geojson',
        output_name='tech_businesses.geojson',
        filter_keywords=['tech', 'software', 'digital', 'computer'],
        filter_columns=['businesstype', 'businessname']
    )
    
    print("Added tech business filter configuration")
    
    print("\nProcessed files saved to: data/processed/")
    print("Summary file: data/processed/data_collection_summary.json")
    print("\nThe DataCollector class now supports:")
    print("- Custom data configurations")
    print("- Multiple file types (CSV, GeoJSON, JSON)")
    print("- Automatic encoding detection")
    print("- Flexible filtering and transformations")
    print("- Dynamic data source addition")


if __name__ == "__main__":
    main()
