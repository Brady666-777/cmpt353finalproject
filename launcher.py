#!/usr/bin/env python3
"""
VancouverPy Project Launcher

This script provides a simple interface to run the main components of the project.
Usage: python launcher.py [option]

Options:
  collect    - Run data collection script
  process    - Run data processing and feature engineering
  train      - Run Python model training script
  notebook   - Launch Jupyter notebook for interactive analysis
  help       - Show this help message
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

def show_help():
    """Show help message"""
    print(__doc__)
    
    print("\nProject Structure:")
    print("├── src/                     # Data processing scripts")
    print("├── data/raw/               # Raw data files")
    print("├── data/processed/         # Processed data files")
    print("├── notebooks/              # Jupyter notebooks")
    print("└── reports/                # Analysis reports")
    
    print("\nQuick Start:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. python launcher.py collect")
    print("3. python launcher.py process        # Standard processing")
    print("   OR python launcher.py spark       # PySpark for large datasets")
    print("4. python launcher.py train")
    print("5. python launcher.py notebook       # Optional: for interactive analysis")
    
    print("\nProcessing Options:")
    print("- process      : Standard pandas processing (smaller datasets)")
    print("- spark        : PySpark processing (large datasets, better performance)")
    
    print("\nDocumentation:")
    print("- README.md: Complete project overview")
    print("- reports/Project_Report.md: Detailed analysis report")
    print("- .env.example: Environment variables template")

def main():
    """Main launcher function"""
    print("VancouverPy: Restaurant Success Prediction")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "collect":
        run_data_collection()
    elif command == "process":
        run_data_processing()
    elif command == "process-spark" or command == "spark":
        run_spark_processing()
    elif command == "train":
        run_model_training()
    elif command == "notebook":
        launch_notebook()
    elif command == "help":
        show_help()
    elif command == "full":
        # Run complete pipeline
        print("Running complete pipeline...")
        if run_data_collection():
            if run_data_processing():
                if run_model_training():
                    print("Pipeline completed! Launching notebook for interactive analysis...")
                    launch_notebook()
    else:
        print(f"Unknown command: {command}")
        print("Use 'python launcher.py help' to see available options")

if __name__ == "__main__":
    main()
