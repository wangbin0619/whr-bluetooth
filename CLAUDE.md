# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
uv sync                    # Install dependencies using uv package manager
```

### Running Applications
```bash
uv run python BluetoothLocator-main/main.py      # Run the main Bluetooth locator GUI
uv run python BluetoothLocator-main/n_calculate.py  # Run path loss exponent calculation utility
```

### Building Executable
```bash
cd BluetoothLocator-main
build.bat                  # Build Windows executable using PyInstaller
# or manually:
uv run pyinstaller --clean bluetooth_locator.spec --noconfirm
```

## Architecture Overview

This is a Bluetooth beacon indoor positioning system with two main components:

### Main Application (main.py)
- **MQTT Integration**: Subscribes to `/device/blueTooth/station/+` topics to receive beacon data
- **Data Format**: Processes semicolon-separated strings: `MAC,RSSI,angle;MAC,RSSI,angle;...;deviceID`
- **Real-time Positioning**: Uses RSSI-based trilateration with configurable path loss model
- **GUI Interface**: Tkinter-based interface with live visualization and beacon management
- **Data Persistence**: Saves raw data to `bluetooth_position_data.csv` and calculated positions to `terminal_locations.csv`

### Core Classes
- **ConfigManager**: YAML configuration management for MQTT and RSSI model parameters
- **BeaconLocationCalculator**: Implements positioning algorithms using free-space path loss model
- **MQTTHandler**: Manages MQTT client connection and message processing
- **BluetoothLocatorGUI**: Main interface with real-time plotting capabilities

### Positioning Algorithm
Uses path loss model: `RSSI = tx_power - 10 * n * log10(distance)`
- Configurable via `config.yaml`: `tx_power` (default: -60 dBm), `path_loss_exponent` (default: 2.0)
- Trilateration using least squares method for 3+ beacons
- Weighted centroid calculation for multiple beacon scenarios

### Utility Tool (n_calculate.py)
Standalone tool for calculating optimal path loss exponent from collected RSSI/distance data in `rssi_filtered.xlsx`. Outputs statistical analysis and visualization of path loss parameter distribution.

### Data Files
- `config.yaml`: MQTT broker settings and RSSI model configuration
- `beacon_database.json`: Beacon MAC address to coordinate mapping
- `bluetooth_position_data.csv`: Raw MQTT message storage
- `terminal_locations.csv`: Calculated position results with timestamps