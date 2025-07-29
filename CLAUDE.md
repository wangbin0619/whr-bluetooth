# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bluetooth indoor positioning research project containing two main components:

1. **BluetoothLocator-main**: A Python-based real-time Bluetooth positioning system using MQTT, RSSI-based trilateration algorithms, and a Tkinter GUI
2. **IndoorPos-master**: A Java-based indoor positioning server system using Spring framework, RMI, and Netty for high-concurrency positioning services

## Development Commands

### Python Project (BluetoothLocator-main)
```bash
# Navigate to the Python project directory
cd BluetoothLocator-main

# Sync dependencies using uv
uv sync

# Run the main application
uv run python main.py

# Generate synthetic test data
uv run python get_data.py

# Calculate path loss exponent from measurement data
cd tools
uv run python n_calculate.py

# Build executable
uv run pyinstaller --clean bluetooth_locator.spec --noconfirm
```

### Java Project (IndoorPos-master)
```bash
# Navigate to the Java project directory  
cd IndoorPos-master

# Compile and package with Maven
mvn compile
mvn package

# Run tests
mvn test

# Run the positioning server (requires database setup)
java -cp target/classes org.hqu.indoor_pos.server.Server
```

## Detailed Component Analysis

### BluetoothLocator-main (Python Component)

#### Core Classes and Architecture
- **ConfigManager**: YAML-based configuration management with default fallbacks for MQTT, RSSI model, and optimization parameters
- **BeaconLocationCalculator**: Main positioning engine implementing multiple algorithms:
  - Gradient descent optimization with multi-start capability
  - RSSI-to-distance conversion using both default and improved linear models
  - Haversine distance calculations for geographic coordinates
  - Beacon database management with JSON persistence
- **MQTTClient**: Real-time data processing from MQTT broker with threaded message handling
- **GUI Components**: Tkinter-based interface for real-time visualization and configuration

#### Key Features
- **Multi-algorithm positioning**: Supports gradient descent with configurable multi-start optimization
- **Flexible RSSI models**: Both standard path loss model and improved linear fitting model
- **Real-time processing**: MQTT-based data ingestion with immediate position calculation
- **Data persistence**: CSV export for both raw Bluetooth data and calculated positions
- **Configuration management**: YAML-based settings with runtime modification capabilities
- **Synthetic data generation**: Built-in test data generator using Haversine interpolation

#### Distance Calculation Models
1. **Default model**: `distance = 10^((TxPower - RSSI) / (10 * n))`
2. **Improved model**: `distance = (RSSI + b) / a` (linear fitting parameters)

### IndoorPos-master (Java Component)

#### Core Architecture
- **Server.java**: Main Netty-based server with Spring framework integration
  - Uses NIO event loops for high-concurrency handling
  - Maintains cached maps for employees, rooms, base stations, and environment factors
  - Implements blocking queue for location results processing
- **PosServerHandler.java**: Netty channel handler for processing positioning requests
- **Algorithm package**: Multiple positioning implementations using Jama matrix library

#### Positioning Algorithms
1. **Trilateral.java**: Basic three-point trilateration using least squares matrix operations
2. **WeightTrilateral.java**: Distance-weighted algorithm with combinatorial approach
   - Calculates all possible 3-beacon combinations
   - Applies inverse distance weighting: `weight = 1/distance`
   - Aggregates results across all combinations
3. **Centroid.java**: Triangle centroid-based positioning algorithm

#### Data Model Classes
- **BleBase.java**: Beacon representation with RSSI-to-distance conversion
  - Uses path loss model: `distance = 10^((p0-rssi)/(10*n))`
  - Applies height compensation for horizontal distance calculation
- **Location.java**: Positioning result with coordinates, timestamp, and metadata
- **BaseStation.java**, **Employee.java**: Database entity representations

#### Database Integration
- Spring JDBC with connection pooling (Druid)
- Cached lookups for base station coordinates, employee mappings, and environment factors
- Real-time data persistence for positioning results

#### Network Architecture
- **Port 50006**: Main positioning service using Netty NIO
- **DispServer**: Separate thread for client display communication
- Line-based frame decoding for message parsing
- Concurrent processing with thread-safe collections

## Key Technical Components

### RSSI Distance Models
Both systems implement the wireless signal attenuation model with slight variations:
- **Python**: `RSSI = TxPower - 10 * n * log10(distance)` (default: TxPower=-59, n=2.0)
- **Java**: `distance = 10^((p0-rssi)/(10*n))` with height compensation

### Positioning Algorithm Comparison
- **Python**: Uses scipy.optimize with gradient descent and multi-start capability
- **Java**: Matrix-based least squares using Jama library with combinatorial weighting

### Data Flow Architecture
1. **Python System**: MQTT → Real-time processing → GUI visualization → CSV export
2. **Java System**: TCP/Netty → Algorithm processing → Database storage → Client distribution

### Error Handling and Validation
- **Python**: Geographic coordinate validation, distance reasonableness checks, optimization convergence validation
- **Java**: Database connection management, matrix operation error handling, concurrent access protection

## Configuration Files

- **config/config.yaml**: MQTT broker settings, RSSI model parameters, optimization settings
- **beacon/beacon_database*.json**: Beacon coordinate databases with longitude/latitude/altitude
- **config/bluetooth_data.json**: Data processing configuration paths
- **config/pyproject.toml**: Python project dependencies (paho-mqtt, pandas, numpy, matplotlib, scipy)
- **pom.xml**: Java dependencies (Spring, Netty, MySQL, Druid, Jama matrix library)

## Database Schema (Java Component)
- **base_station**: Base station coordinates and room associations
- **employee**: Terminal ID to employee mapping
- **env_factor**: Environment-specific calibration parameters (height, path loss exponent, reference power)
- **location_history**: Stored positioning results with timestamps

## Development Notes

### Python Component
- Uses modern `uv` package manager for dependency management
- Implements thread-safe MQTT message processing with queue-based architecture
- Supports both synthetic data generation and real measurement data processing
- GUI provides real-time visualization with matplotlib integration

### Java Component  
- Spring framework with dependency injection and JDBC template patterns
- High-performance Netty server supporting concurrent client connections
- Matrix operations using established Jama library for numerical stability
- Database connection pooling for production scalability

### Testing and Calibration
- **n_calculate.py**: Statistical analysis tool for path loss exponent calibration from measurement data
- **get_data.py**: Synthetic data generator using Haversine distance calculations
- Excel-based test datasets in `data/` directory for algorithm validation