#!/usr/bin/env python3
"""
Quick Start Script for NWB Data Viewer
This script helps you configure and run the NWB Data Viewer quickly.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print a nice banner for the quick start script"""
    print("=" * 60)
    print("🧠 NWB Data Viewer - Quick Start")
    print("=" * 60)
    print("Interactive Neural Data Visualization Tool")
    print("Designed by: Parviz Ghaderi")
    print("Nature Neuroscience 2025")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("📋 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def configure_data_path():
    """Help user configure the data path"""
    print("\n📁 Configuring data path...")
    
    # Check if NWB_DIR is already set
    current_path = None
    try:
        with open("nwb_data_viewer.py", "r") as f:
            for line_num, line in enumerate(f, 1):
                if "NWB_DIR" in line and "=" in line:
                    current_path = line.split("=")[1].strip().strip("'\"")
                    break
    except FileNotFoundError:
        print("❌ Error: nwb_data_viewer.py not found")
        return False
    
    print(f"Current data path: {current_path}")
    
    # Ask user for new path
    print("\nPlease enter the path to your NWB data files:")
    print("(Press Enter to keep current path)")
    new_path = input("Path: ").strip()
    
    if new_path:
        # Update the path in the file
        try:
            with open("nwb_data_viewer.py", "r") as f:
                content = f.read()
            
            # Replace the NWB_DIR line
            import re
            pattern = r'NWB_DIR\s*=\s*[^\n]+'
            replacement = f"NWB_DIR = r'{new_path}'"
            new_content = re.sub(pattern, replacement, content)
            
            with open("nwb_data_viewer.py", "w") as f:
                f.write(new_content)
            
            print(f"✅ Data path updated to: {new_path}")
        except Exception as e:
            print(f"❌ Error updating data path: {e}")
            return False
    
    return True

def check_data_files():
    """Check if NWB files are available"""
    print("\n🔍 Checking for NWB data files...")
    
    try:
        with open("nwb_data_viewer.py", "r") as f:
            content = f.read()
        
        # Extract NWB_DIR path
        import re
        match = re.search(r'NWB_DIR\s*=\s*r?[\'"]([^\'"]+)[\'"]', content)
        if match:
            data_path = match.group(1)
            
            if os.path.exists(data_path):
                nwb_files = [f for f in os.listdir(data_path) if f.endswith('.nwb')]
                if nwb_files:
                    print(f"✅ Found {len(nwb_files)} NWB files in {data_path}")
                    print(f"   Files: {', '.join(nwb_files[:5])}{'...' if len(nwb_files) > 5 else ''}")
                    return True
                else:
                    print(f"⚠️  No NWB files found in {data_path}")
                    print("   Please ensure your NWB files are in the specified directory")
                    return False
            else:
                print(f"❌ Data path does not exist: {data_path}")
                return False
        else:
            print("❌ Could not find NWB_DIR configuration")
            return False
    except Exception as e:
        print(f"❌ Error checking data files: {e}")
        return False

def run_application():
    """Run the NWB Data Viewer application"""
    print("\n🚀 Starting NWB Data Viewer...")
    print("The application will open in your web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8050")
    print("\nPress Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "nwb_data_viewer.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function for quick start"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Configure data path
    if not configure_data_path():
        return
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Warning: No NWB files found, but you can still run the application")
        print("   You'll need to configure the data path later")
    
    # Run application
    run_application()

if __name__ == "__main__":
    main() 