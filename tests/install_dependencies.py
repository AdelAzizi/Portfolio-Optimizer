#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Analysis System Dependencies Installer

This script installs all required dependencies for the analysis system.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install dependencies from requirements file"""
    
    print("🚀 Starting installation of comprehensive analysis system dependencies...")
    print("=" * 60)
    
    # Requirements file path
    requirements_file = Path(__file__).parent / "requirements_analysis.txt"
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        # Install dependencies
        print(f"📦 Installing dependencies from {requirements_file.name}...")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All dependencies installed successfully!")
            print("\n📋 Installation summary:")
            print(result.stdout.split('\n')[-10:])  # Last 10 lines
            return True
        else:
            print("❌ Error installing dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_critical_packages():
    """Check installation of critical packages"""
    
    print("\n🔍 Checking critical packages...")
    
    critical_packages = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 
        'seaborn', 'scikit-learn', 'yaml', 'jinja2'
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        return False
    else:
        print("\n🎉 All critical packages are installed!")
        return True

def create_virtual_env():
    """Create virtual environment (optional)"""
    
    response = input("\n❓ Do you want to create a virtual environment? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        venv_name = "analysis_env"
        
        try:
            print(f"🔧 Creating virtual environment {venv_name}...")
            subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
            
            print(f"✅ Virtual environment {venv_name} created!")
            print(f"💡 To activate:")
            
            if os.name == 'nt':  # Windows
                print(f"   {venv_name}\\Scripts\\activate")
            else:  # Unix/Linux/Mac
                print(f"   source {venv_name}/bin/activate")
                
            return True
            
        except Exception as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
    
    return True

def main():
    """Main function"""
    
    print("🎯 Comprehensive Analysis System Dependencies Installer")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"⚠️ Your Python version is {python_version.major}.{python_version.minor}")
        print("💡 Python 3.8+ is recommended")
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Suggest creating virtual environment
    create_virtual_env()
    
    # Install dependencies
    if install_requirements():
        # Check installation
        if check_critical_packages():
            print("\n🎉 Installation completed successfully!")
            print("\n🚀 You can now run the analysis system:")
            print("   python run_comprehensive_analysis.py")
        else:
            print("\n⚠️ Some packages are missing. Please try again.")
    else:
        print("\n❌ Installation failed. Please check the errors.")

if __name__ == "__main__":
    main()