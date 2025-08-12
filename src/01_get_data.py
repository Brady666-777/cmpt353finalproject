"""
Data Collection Script for VancouverPy Restaurant Success Prediction

This script handles the collection of data from multiple sources:
- City of Vancouver Open Data Portal (Business Licenses, Local Areas, Traffic)
- Statistics Canada (Census Data)
- Yelp Fusion API (Restaurant Reviews and Ratings)

Author: VancouverPy Project Team
Date: August 2025
"""

import os
import requests
import pandas as pd
import geopandas as gpd
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Main class for collecting data from various sources"""
    
    def __init__(self):
        self.vancouver_api_key = os.getenv('VANCOUVER_API_KEY')
        self.yelp_api_key = os.getenv('YELP_API_KEY')
        self.data_dir = '../data/raw'
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # API endpoints
        self.vancouver_base_url = "https://opendata.vancouver.ca/api/records/1.0"
        self.yelp_base_url = "https://api.yelp.com/v3"
        
    def get_business_licenses(self) -> pd.DataFrame:
        """
        Fetch business license data from City of Vancouver Open Data Portal
        Focus on food service establishments
        """
        logger.info("Fetching business license data...")
        
        # Business license dataset ID
        dataset_id = "business-licences"
        
        params = {
            'dataset': dataset_id,
            'rows': 10000,  # Adjust based on dataset size
            'q': 'food OR restaurant OR cafe OR bakery',  # Filter for food establishments
            'facet': ['businesstype', 'status', 'localarea']
        }
        
        if self.vancouver_api_key:
            params['api_key'] = self.vancouver_api_key
            
        try:
            response = requests.get(f"{self.vancouver_base_url}/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            records = [record['fields'] for record in data.get('records', [])]
            
            df = pd.DataFrame(records)
            df.to_csv(f"{self.data_dir}/business_licenses.csv", index=False)
            
            logger.info(f"Collected {len(df)} business license records")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching business licenses: {e}")
            return pd.DataFrame()
    
    def get_local_areas(self) -> gpd.GeoDataFrame:
        """
        Fetch Vancouver local area boundaries (22 neighborhoods)
        """
        logger.info("Fetching local area boundaries...")
        
        dataset_id = "local-area-boundary"
        
        params = {
            'dataset': dataset_id,
            'rows': 25,  # Vancouver has 22 local areas
            'format': 'geojson'
        }
        
        if self.vancouver_api_key:
            params['api_key'] = self.vancouver_api_key
            
        try:
            response = requests.get(f"{self.vancouver_base_url}/search", params=params)
            response.raise_for_status()
            
            # Save raw GeoJSON
            with open(f"{self.data_dir}/local_areas.geojson", 'w') as f:
                f.write(response.text)
            
            # Load as GeoDataFrame
            gdf = gpd.read_file(f"{self.data_dir}/local_areas.geojson")
            
            logger.info(f"Collected {len(gdf)} local area boundaries")
            return gdf
            
        except requests.RequestException as e:
            logger.error(f"Error fetching local areas: {e}")
            return gpd.GeoDataFrame()
    
    def get_traffic_counts(self) -> pd.DataFrame:
        """
        Fetch traffic count data for mobility analysis
        """
        logger.info("Fetching traffic count data...")
        
        dataset_id = "traffic-signal-counts"
        
        params = {
            'dataset': dataset_id,
            'rows': 5000,
            'facet': ['year']
        }
        
        if self.vancouver_api_key:
            params['api_key'] = self.vancouver_api_key
            
        try:
            response = requests.get(f"{self.vancouver_base_url}/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            records = [record['fields'] for record in data.get('records', [])]
            
            df = pd.DataFrame(records)
            df.to_csv(f"{self.data_dir}/traffic_counts.csv", index=False)
            
            logger.info(f"Collected {len(df)} traffic count records")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching traffic counts: {e}")
            return pd.DataFrame()
    
    def get_yelp_restaurants(self, location: str = "Vancouver, BC", limit: int = 1000) -> pd.DataFrame:
        """
        Fetch restaurant data from Yelp Fusion API
        """
        logger.info("Fetching restaurant data from Yelp...")
        
        if not self.yelp_api_key:
            logger.error("Yelp API key not found. Please set YELP_API_KEY environment variable.")
            return pd.DataFrame()
        
        headers = {
            'Authorization': f'Bearer {self.yelp_api_key}'
        }
        
        restaurants = []
        offset = 0
        
        while len(restaurants) < limit:
            params = {
                'location': location,
                'categories': 'restaurants',
                'limit': min(50, limit - len(restaurants)),  # Yelp API limit is 50 per request
                'offset': offset
            }
            
            try:
                response = requests.get(f"{self.yelp_base_url}/businesses/search", 
                                      headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                businesses = data.get('businesses', [])
                
                if not businesses:
                    break
                    
                restaurants.extend(businesses)
                offset += len(businesses)
                
                # Rate limiting - Yelp allows 5000 requests per day
                time.sleep(0.1)
                
                logger.info(f"Collected {len(restaurants)} restaurants so far...")
                
            except requests.RequestException as e:
                logger.error(f"Error fetching Yelp data: {e}")
                break
        
        df = pd.DataFrame(restaurants)
        if not df.empty:
            df.to_csv(f"{self.data_dir}/yelp_restaurants.csv", index=False)
            
        logger.info(f"Total restaurants collected: {len(df)}")
        return df
    
    def get_census_data(self) -> pd.DataFrame:
        """
        Placeholder for Statistics Canada census data
        In practice, you would download from Statistics Canada's API or website
        """
        logger.info("Census data collection placeholder - implement based on specific requirements")
        
        # Create a placeholder CSV with expected structure
        placeholder_data = {
            'local_area': ['Downtown', 'West End', 'Kitsilano'],  # Sample areas
            'population': [46000, 42000, 41000],
            'median_income': [65000, 75000, 85000],
            'age_median': [35, 38, 40]
        }
        
        df = pd.DataFrame(placeholder_data)
        df.to_csv(f"{self.data_dir}/census_data.csv", index=False)
        
        return df
    
    def get_transit_data(self) -> pd.DataFrame:
        """
        Placeholder for public transit data
        In practice, you could use other transit data sources or APIs
        """
        logger.info("Transit data collection placeholder - implement based on available sources")
        
        # Create placeholder for transit stops
        placeholder_data = {
            'stop_id': ['1001', '1002', '1003'],
            'stop_name': ['Granville Station', 'Burrard Station', 'Broadway-City Hall'],
            'stop_lat': [49.2827, 49.2858, 49.2634],
            'stop_lon': [-123.1207, -123.1207, -123.1157]
        }
        
        df = pd.DataFrame(placeholder_data)
        df.to_csv(f"{self.data_dir}/transit_stops.csv", index=False)
        
        return df
    
    def collect_all_data(self):
        """
        Main method to collect all required datasets
        """
        logger.info("Starting comprehensive data collection...")
        
        # Collect all datasets
        self.get_business_licenses()
        self.get_local_areas()
        self.get_traffic_counts()
        self.get_yelp_restaurants()
        self.get_census_data()
        self.get_transit_data()
        
        logger.info("Data collection completed!")


def main():
    """
    Main execution function
    """
    collector = DataCollector()
    collector.collect_all_data()


if __name__ == "__main__":
    main()
