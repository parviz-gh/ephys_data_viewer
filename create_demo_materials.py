#!/usr/bin/env python3
"""
Demo Materials Creation Script
This script helps create screenshots and provides guidance for video recording.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def print_demo_banner():
    """Print banner for demo materials creation"""
    print("=" * 70)
    print("🎬 NWB Data Viewer - Demo Materials Creation")
    print("=" * 70)
    print("This script helps you create screenshots and demo videos")
    print("for your GitHub repository.")
    print("=" * 70)

def check_screenshots_directory():
    """Ensure screenshots directory exists"""
    screenshots_dir = Path("screenshots")
    if not screenshots_dir.exists():
        screenshots_dir.mkdir()
        print("✅ Created screenshots directory")
    else:
        print("✅ Screenshots directory exists")
    return screenshots_dir

def provide_screenshot_instructions():
    """Provide detailed instructions for taking screenshots"""
    print("\n📸 SCREENSHOT INSTRUCTIONS")
    print("-" * 50)
    
    screenshots_needed = [
        {
            "name": "main_interface.png",
            "description": "Main application interface showing file selection and controls",
            "instructions": [
                "1. Start the NWB Data Viewer application",
                "2. Load a sample NWB file",
                "3. Take a screenshot of the entire interface",
                "4. Save as 'main_interface.png' in screenshots/ folder"
            ]
        },
        {
            "name": "3d_brain.png",
            "description": "3D brain visualization with units plotted",
            "instructions": [
                "1. Navigate to the 3D brain tab/plot",
                "2. Rotate the brain to show units clearly",
                "3. Ensure Allen CCF mesh is visible",
                "4. Take a screenshot and save as '3d_brain.png'"
            ]
        },
        {
            "name": "psth_analysis.png",
            "description": "PSTH analysis results showing neural and behavioral data",
            "instructions": [
                "1. Generate a PSTH plot with some data",
                "2. Show both neural and behavioral PSTH",
                "3. Ensure plots are clear and well-labeled",
                "4. Take a screenshot and save as 'psth_analysis.png'"
            ]
        },
        {
            "name": "video_thumbnail.png",
            "description": "Eye-catching thumbnail for the demo video",
            "instructions": [
                "1. Create a composite image showing key features",
                "2. Include text overlay: 'NWB Data Viewer'",
                "3. Make it visually appealing and professional",
                "4. Save as 'video_thumbnail.png' (1280x720 recommended)"
            ]
        }
    ]
    
    for i, screenshot in enumerate(screenshots_needed, 1):
        print(f"\n{i}. {screenshot['name']}")
        print(f"   Description: {screenshot['description']}")
        print("   Instructions:")
        for instruction in screenshot['instructions']:
            print(f"   {instruction}")
    
    return screenshots_needed

def provide_video_script():
    """Provide a script for recording the demo video"""
    print("\n🎥 VIDEO RECORDING SCRIPT")
    print("-" * 50)
    
    script = [
        {
            "section": "Introduction (0:00-0:15)",
            "content": [
                "Welcome to the NWB Data Viewer",
                "Interactive tool for neural data visualization",
                "Designed for contextual gating studies"
            ]
        },
        {
            "section": "Application Overview (0:15-0:30)",
            "content": [
                "Show the main interface",
                "Explain file selection and controls",
                "Highlight the clean, professional design"
            ]
        },
        {
            "section": "Data Loading (0:30-0:45)",
            "content": [
                "Demonstrate loading an NWB file",
                "Show metadata and trial information",
                "Explain the data structure"
            ]
        },
        {
            "section": "3D Brain Visualization (0:45-1:15)",
            "content": [
                "Navigate to 3D brain view",
                "Rotate and zoom the brain",
                "Show units plotted at real locations",
                "Demonstrate clicking on units",
                "Show activity-based coloring"
            ]
        },
        {
            "section": "PSTH Analysis (1:15-1:45)",
            "content": [
                "Generate PSTH plots",
                "Show neural responses",
                "Demonstrate behavioral correlation",
                "Apply trial filters",
                "Show real-time updates"
            ]
        },
        {
            "section": "Advanced Features (1:45-2:15)",
            "content": [
                "Switch between analysis modes",
                "Show brain region comparison",
                "Demonstrate custom time windows",
                "Show export capabilities"
            ]
        },
        {
            "section": "Conclusion (2:15-2:30)",
            "content": [
                "Summarize key features",
                "Mention research applications",
                "Provide contact information"
            ]
        }
    ]
    
    for section in script:
        print(f"\n{section['section']}")
        for content in section['content']:
            print(f"  • {content}")
    
    return script

def provide_recording_tips():
    """Provide tips for high-quality video recording"""
    print("\n💡 RECORDING TIPS")
    print("-" * 50)
    
    tips = [
        "🎯 **Preparation**",
        "  • Close unnecessary applications",
        "  • Ensure good lighting",
        "  • Test your microphone",
        "  • Have a script ready",
        "",
        "🎬 **Recording Settings**",
        "  • Resolution: 1920x1080 (Full HD)",
        "  • Frame rate: 30 fps",
        "  • Audio: Clear narration",
        "  • Format: MP4",
        "",
        "⚡ **Performance**",
        "  • Use a powerful computer",
        "  • Close other programs",
        "  • Test with sample data first",
        "  • Have backup plans ready",
        "",
        "🎨 **Visual Quality**",
        "  • Use high DPI displays",
        "  • Ensure good contrast",
        "  • Keep interface clean",
        "  • Show smooth interactions",
        "",
        "🔊 **Audio Quality**",
        "  • Use a good microphone",
        "  • Speak clearly and slowly",
        "  • Avoid background noise",
        "  • Consider adding captions"
    ]
    
    for tip in tips:
        print(tip)

def suggest_recording_tools():
    """Suggest recording tools for different platforms"""
    print("\n🛠️ RECORDING TOOLS")
    print("-" * 50)
    
    tools = {
        "Windows": [
            "OBS Studio (Free, professional)",
            "Windows Game Bar (Win + G, built-in)",
            "Camtasia (Paid, easy to use)",
            "Bandicam (Paid, lightweight)"
        ],
        "macOS": [
            "QuickTime Player (Built-in)",
            "OBS Studio (Free, professional)",
            "ScreenFlow (Paid, Mac-specific)",
            "Camtasia (Paid, easy to use)"
        ],
        "Linux": [
            "OBS Studio (Free, professional)",
            "SimpleScreenRecorder (Free, lightweight)",
            "Kazam (Free, simple)",
            "RecordMyDesktop (Free, basic)"
        ]
    }
    
    for platform, tool_list in tools.items():
        print(f"\n{platform}:")
        for tool in tool_list:
            print(f"  • {tool}")

def create_github_workflow():
    """Create a GitHub workflow for automated testing"""
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: Test NWB Data Viewer

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run basic tests
      run: |
        python -c "import dash, plotly, numpy, pandas, scipy, h5py, pynwb; print('All imports successful')"
    
    - name: Check code syntax
      run: |
        python -m py_compile nwb_data_viewer.py
        echo "Syntax check passed"
"""
    
    workflow_file = workflow_dir / "test.yml"
    with open(workflow_file, "w") as f:
        f.write(workflow_content)
    
    print(f"✅ Created GitHub workflow: {workflow_file}")

def main():
    """Main function for demo materials creation"""
    print_demo_banner()
    
    # Check and create directories
    screenshots_dir = check_screenshots_directory()
    
    # Provide instructions
    provide_screenshot_instructions()
    provide_video_script()
    provide_recording_tips()
    suggest_recording_tools()
    
    # Create GitHub workflow
    create_github_workflow()
    
    print("\n" + "=" * 70)
    print("🎉 Demo materials creation guide complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Take the required screenshots")
    print("2. Record your demo video")
    print("3. Upload to YouTube or similar platform")
    print("4. Update README.md with actual links")
    print("5. Push to GitHub")
    print("\nGood luck with your repository! 🚀")

if __name__ == "__main__":
    main() 