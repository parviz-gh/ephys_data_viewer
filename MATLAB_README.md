# 🧠 MATLAB Data Viewer

> **MATLAB Version of the Interactive Neural Data Visualization Tool**

This is the MATLAB implementation of the NWB Data Viewer, providing similar functionality to the Python version but using MATLAB's native plotting and GUI capabilities.

## 📋 Requirements

- **MATLAB R2020b or higher**
- **Statistics and Machine Learning Toolbox**
- **Signal Processing Toolbox**
- **Image Processing Toolbox** (for 3D visualization)

## 🚀 Installation

1. **Download the files**
   - `DataViewer.mlapp` - Main application file
   - `HistologyFunction_helper/` - Helper functions and data

2. **Add to MATLAB path**
   ```matlab
   addpath('path/to/DataViewer');
   addpath('path/to/DataViewer/HistologyFunction_helper');
   ```

3. **Run the application**
   ```matlab
   DataViewer
   ```

## 🎯 Features

### **Interactive GUI**
- **File Selection**: Browse and load NWB files
- **Data Preview**: View file structure and metadata
- **Real-time Plotting**: Interactive plots with zoom and pan

### **Neural Data Analysis**
- **PSTH Calculation**: Peristimulus time histograms
- **Unit Selection**: Choose individual neurons or brain regions
- **Trial Filtering**: Filter by behavioral conditions
- **Statistical Analysis**: Mean, SEM, and significance testing

### **3D Brain Visualization**
- **Allen CCF Integration**: Brain atlas overlay
- **Unit Positioning**: Plot units at actual recording locations
- **Interactive 3D**: Rotate, zoom, and explore brain regions

### **Behavioral Correlation**
- **Movement Tracking**: Whisker, jaw, and tongue movements
- **Signal Processing**: Filtering and speed calculations
- **Temporal Alignment**: Align behavioral data with neural events

## 📁 Data Format

The MATLAB version expects the same NWB data structure as the Python version:

```
nwbfile/
├── units/
│   ├── spike_times
│   ├── location (brain region)
│   └── ccf_xyz (coordinates)
├── trials/
│   ├── start_time
│   ├── lick_flag
│   ├── trial_type
│   └── early_lick
└── processing/
    └── behavior/
        ├── BehavioralTimeSeries/
        │   ├── C2Whisker_Angle
        │   ├── Jaw_Coordinate
        │   └── Piezo_lick_trace
        └── BehavioralEvents/
```

## 🎮 Usage Guide

### 1. **Load Data**
- Click "Browse" to select an NWB file
- View file information in the metadata panel

### 2. **Configure Analysis**
- Select event field for PSTH alignment
- Choose behavioral signal for correlation
- Set trial filters if needed

### 3. **Generate Plots**
- Click "Plot PSTH" to generate neural response plots
- Use "Plot Behavioral" for movement data
- Explore 3D brain visualization

### 4. **Export Results**
- Save plots as high-resolution images
- Export data for further analysis
- Generate publication-ready figures

## 🔧 Configuration

### Data Path
Update the data directory in the application:
```matlab
% In DataViewer.mlapp, find and update:
dataPath = 'path/to/your/nwb/files';
```

### Brain Atlas
The application uses Allen CCF coordinates. Ensure the brain atlas data is available:
```matlab
% Check if brain atlas is loaded
if ~exist('allenCCF', 'var')
    load('allenCCF.mat');  % Load brain atlas data
end
```

## 📊 Output Formats

### Plots
- **PSTH Plots**: PNG, PDF, EPS formats
- **3D Brain**: Interactive figure with export options
- **Behavioral Data**: Time series plots with statistics

### Data Export
- **Spike Times**: CSV or MAT format
- **PSTH Data**: Structured arrays with statistics
- **Behavioral Signals**: Filtered and processed time series

## 🛠️ Troubleshooting

### Common Issues

1. **File Not Found**
   - Ensure NWB files are in the correct directory
   - Check file permissions

2. **Memory Issues**
   - Close other MATLAB applications
   - Use smaller data subsets for testing

3. **Plotting Errors**
   - Verify data format matches expected structure
   - Check for NaN or missing values

### Performance Tips

- **Large Files**: Use data subsetting for initial exploration
- **3D Visualization**: Reduce mesh resolution for better performance
- **Memory Management**: Clear variables when switching between files

## 🔬 Research Applications

This MATLAB version is particularly useful for:

- **Quick Data Exploration**: Rapid visualization of neural responses
- **Publication Figures**: High-quality plot generation
- **Teaching**: Interactive demonstration of neural data analysis
- **Collaboration**: Sharing with MATLAB-based research groups

## 📚 References

- **Allen Brain Atlas**: [https://mouse.brain-map.org/](https://mouse.brain-map.org/)
- **NWB Format**: [https://www.nwb.org/](https://www.nwb.org/)
- **MATLAB Documentation**: [https://www.mathworks.com/help/](https://www.mathworks.com/help/)

## 🤝 Support

For issues specific to the MATLAB version:
- Check the MATLAB documentation
- Verify toolbox availability
- Test with sample data first

## 📄 License

Same MIT license as the Python version. See [LICENSE](LICENSE) for details.

---

**Note**: The MATLAB version provides similar functionality to the Python version but may have slight differences in interface and performance characteristics. 