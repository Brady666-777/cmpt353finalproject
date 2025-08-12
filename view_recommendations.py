"""
Quick viewer for the restaurant recommendations GeoJSON
"""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import json

def main():
    """View the generated restaurant recommendations"""
    
    print("🍽️  Vancouver Restaurant Location Recommendations")
    print("=" * 50)
    
    # Load the GeoJSON recommendations
    try:
        gdf = gpd.read_file('data/processed/recommended_spots.geojson')
        print(f"📍 Loaded {len(gdf)} recommendations for {gdf['cuisine_type'].nunique()} cuisine types")
        
        # Show summary by cuisine
        cuisine_summary = gdf.groupby('cuisine_type').agg({
            'predicted_success_score': ['count', 'mean', 'max'],
            'competitor_count': 'mean'
        }).round(3)
        
        print("\n🎯 Top Recommendations by Cuisine:")
        top_recs = gdf.groupby('cuisine_type')['predicted_success_score'].max().sort_values(ascending=False).head(10)
        for cuisine, score in top_recs.items():
            print(f"  • {cuisine}: {score:.3f}")
        
        # Load summary JSON for detailed info
        with open('data/processed/recommendation_summary.json', 'r') as f:
            summary = json.load(f)
        
        print(f"\n📊 Analysis Overview:")
        print(f"  • Total cuisines analyzed: {summary['total_cuisines_analyzed']}")
        print(f"  • Total recommendations: {summary['total_recommendations']}")
        print(f"  • Analysis date: {summary['analysis_date'][:10]}")
        
        # Show best locations
        print(f"\n🏆 Best Overall Locations:")
        best_spots = gdf.nlargest(5, 'predicted_success_score')[['cuisine_type', 'predicted_success_score', 'recommendation_reason']]
        for _, row in best_spots.iterrows():
            print(f"  • {row['cuisine_type']}: {row['predicted_success_score']:.3f}")
            print(f"    {row['recommendation_reason']}")
        
        # Create simple plot
        print(f"\n📈 Creating visualization...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot all recommendations with different colors by cuisine type
        cuisines = gdf['cuisine_type'].unique()
        colors = plt.cm.Set3(range(len(cuisines)))
        
        for i, cuisine in enumerate(cuisines[:10]):  # Show top 10 cuisines
            cuisine_data = gdf[gdf['cuisine_type'] == cuisine]
            ax.scatter(cuisine_data.geometry.x, cuisine_data.geometry.y, 
                      c=[colors[i]], label=cuisine, s=60, alpha=0.7)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Vancouver Restaurant Recommendations by Cuisine\n(Top 10 Cuisines)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/recommendations_map.png', dpi=300, bbox_inches='tight')
        print(f"📁 Map saved to: data/processed/recommendations_map.png")
        
        # Show coordinate ranges
        print(f"\n🗺️  Coverage Area:")
        print(f"  • Latitude: {gdf.geometry.y.min():.4f} to {gdf.geometry.y.max():.4f}")
        print(f"  • Longitude: {gdf.geometry.x.min():.4f} to {gdf.geometry.x.max():.4f}")
        
        print(f"\n✅ Analysis complete! Check the GeoJSON file for detailed location data.")
        
    except Exception as e:
        print(f"❌ Error loading recommendations: {e}")

if __name__ == "__main__":
    main()
