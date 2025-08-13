#!/usr/bin/env python3
"""
VancouverPy Project Launcher - REFINED VERSION

This script provides a simple interface to run the main components of the project.
Usage: python launcher.py [option]

Options:
  full       - Run complete pipeline (collect -> process -> train -> validate)
  collect    - Run data collection script (without Yelp dependency)
  process    - Run data processing and feature engineering
  train      - Run refined model training script (3 models, no overfitting)
  validate   - Validate pipeline results
  notebook   - Launch Jupyter notebook for interactive analysis
  help       - Show this help message

Recent Improvements:
- Removed Yelp API dependency
- Streamlined model training (3 models instead of 6)
- Fixed overfitting issues
- Enhanced heatmap visualization
- Better performance metrics
"""

import sys
import os
import subprocess
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

def run_data_collection():
    """Run the data collection script"""
    script_path = SRC_DIR / "01_get_data.py"
    print(f"Running data collection script: {script_path}")
    
    # Check if .env file exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("Warning: .env file not found!")
        print("   Please copy .env.example to .env and add your API keys")
        print("   The script will run but may not collect real data without API keys.")
        print()
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
        print("Data collection completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running data collection: {e}")
        return False
    return True

def run_data_processing():
    """Run the data processing script"""
    script_path = SRC_DIR / "02_clean_and_feature_engineer_real.py"
    print(f"Running data processing script: {script_path}")
    
    # Check if raw data exists
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_data_dir.exists() or not any(raw_data_dir.iterdir()):
        print("Warning: No raw data found!")
        print("   Please run data collection first: python launcher.py collect")
        print()
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
        print("Data processing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running data processing: {e}")
        return False
    return True

def run_spark_processing():
    """Run the PySpark data processing script for large datasets"""
    script_path = SRC_DIR / "02_clean_and_feature_engineer_spark.py"
    print(f"Running PySpark data processing script: {script_path}")
    
    # Check if raw data exists
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_data_dir.exists() or not any(raw_data_dir.iterdir()):
        print("Warning: No raw data found!")
        print("   Please run data collection first: python launcher.py collect")
        print()
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
        print("PySpark data processing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running PySpark data processing: {e}")
        return False
    return True

def run_model_training():
    """Run the Python model training script"""
    script_path = SRC_DIR / "03_model_training.py"
    print(f"Running model training script: {script_path}")
    
    # Check if processed data exists
    processed_data_dir = PROJECT_ROOT / "data" / "processed"
    if not processed_data_dir.exists() or not any(processed_data_dir.iterdir()):
        print("Warning: No processed data found!")
        print("   Please run data processing first: python launcher.py process")
        print()
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
        print("Model training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running model training: {e}")
        return False
    return True

def launch_notebook():
    """Launch Jupyter notebook"""
    notebook_path = NOTEBOOKS_DIR / "model_training.ipynb"
    print(f"Launching Jupyter notebook: {notebook_path}")
    
    try:
        # Try to launch with jupyter
        subprocess.run([
            sys.executable, "-m", "jupyter", "notebook", 
            str(notebook_path)
        ], cwd=PROJECT_ROOT)
    except FileNotFoundError:
        print("Jupyter not found. Installing jupyter...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "jupyter"], check=True)
            subprocess.run([
                sys.executable, "-m", "jupyter", "notebook", 
                str(notebook_path)
            ], cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as e:
            print(f"Error launching notebook: {e}")
            return False
    return True

def run_validation():
    """Run pipeline validation"""
    script_path = PROJECT_ROOT / "validate_pipeline.py"
    print(f"Running pipeline validation: {script_path}")
    
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)
        print("Pipeline validation completed!")
    except subprocess.CalledProcessError as e:
        print(f"Error running validation: {e}")
        return False
    return True

def run_full_pipeline():
    """Run complete pipeline with validation"""
    print("🚀 RUNNING COMPLETE PIPELINE")
    print("=" * 50)
    
    print("\n📊 Step 1: Data Collection")
    if not run_data_collection():
        return False
    
    print("\n🔧 Step 2: Data Processing & Feature Engineering")
    if not run_data_processing():
        return False
    
    print("\n🤖 Step 3: Model Training (Refined)")
    if not run_model_training():
        return False
    
    print("\n✅ Step 4: Pipeline Validation")
    if not run_validation():
        return False
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("🔗 All components working correctly")
    print("📈 Models trained and ready for predictions")
    return True

def show_help():
    """Show help message"""
    print(__doc__)
    
    print("\nProject Structure:")
    print("├── src/                     # Data processing scripts")
    print("├── data/raw/               # Raw data files")
    print("├── data/processed/         # Processed data files")
    print("├── notebooks/              # Jupyter notebooks")
    print("└── reports/                # Analysis reports")
    
    print("\nQuick Start (REFINED PIPELINE):")
    print("1. python launcher.py full           # Complete pipeline")
    print("   OR run individual steps:")
    print("2. python launcher.py collect        # Data collection (no Yelp)")
    print("3. python launcher.py process        # Feature engineering")  
    print("4. python launcher.py train          # Model training (3 models)")
    print("5. python launcher.py validate       # Validation check")
    print("6. python launcher.py notebook       # Interactive analysis")
    
    print("\nPipeline Improvements:")
    print("- ✅ Removed Yelp API dependency")
    print("- ✅ Streamlined to 3 relevant models")  
    print("- ✅ Fixed overfitting issues")
    print("- ✅ Enhanced heatmap visualization")
    print("- ✅ Better performance (R2=0.355)")
    
    print("\nProcessing Options:")
    print("- process      : Standard pandas processing (recommended)")
    print("- spark        : PySpark processing (for large datasets)")
    
    print("\nDocumentation:")
    print("- README.md: Complete project overview")
    print("- REFINEMENT_SUMMARY.md: Latest improvements") 
    print("- MODEL_IMPROVEMENTS_SUMMARY.md: Model details")

def main():
    """Main launcher function"""
    print("VancouverPy: Restaurant Success Prediction")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "full":
        run_full_pipeline()
    elif command == "collect":
        run_data_collection()
    elif command == "process":
        run_data_processing()
    elif command == "process-spark" or command == "spark":
        run_spark_processing()
    elif command == "train":
        run_model_training()
    elif command == "validate":
        run_validation()
    elif command == "notebook":
        launch_notebook()
    elif command == "help":
        show_help()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: full, collect, process, train, validate, notebook, help")
        print("Use 'python launcher.py help' to see available options")

if __name__ == "__main__":
    main()
