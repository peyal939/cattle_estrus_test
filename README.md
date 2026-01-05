# 🐄 Cattle Estrus Detection System

A comprehensive IoT-based cattle estrus detection system that uses sensor data (accelerometer, gyroscope, magnetometer) to monitor cattle activity patterns and detect estrus (heat) cycles in real-time.

## 📋 Overview

This system analyzes cattle behavior through wearable sensors and applies machine learning techniques to:
- Monitor daily activity patterns (resting, eating, ruminating, walking)
- Detect estrus cycles through multi-condition analysis
- Provide real-time updates via WebSocket connections
- Store and visualize historical data
- Support multi-user authentication and management

## 🚀 Features

### Core Functionality
- **Real-time Sensor Processing**: Processes accelerometer, gyroscope, and magnetometer data
- **Activity Classification**: Automatically classifies cattle behavior into 4 states:
  - Resting
  - Eating
  - Ruminating
  - Walking
- **Estrus Detection**: Advanced multi-condition algorithm using:
  - PCA-based activity scoring
  - Rolling baseline comparison
  - Relative and absolute activity thresholds
  - Walking fraction analysis
  - Dominance ratio validation

### Web Application
- **FastAPI Backend**: High-performance async API
- **MongoDB Integration**: Scalable data storage
- **Redis Caching**: Optimized query performance
- **WebSocket Support**: Real-time data streaming
- **User Authentication**: Secure JWT-based authentication
- **Interactive Dashboard**: Web-based visualization and monitoring
- **Admin Panel**: User management and system configuration

### Data Processing
- **Daily Metrics Computation**: Automated aggregation of sensor data
- **PCA Analysis**: Dimensionality reduction for activity scoring
- **Dynamic Thresholds**: Adaptive activity classification based on rolling windows
- **Robust Estrus Detection**: Multi-condition validation to reduce false positives

## 📦 Installation

### Prerequisites
- Python 3.8+
- MongoDB
- Redis
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/peyal939/cattle_estrus_test.git
   cd cattle_estrus_test
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   MONGO_URI=mongodb://localhost:27017
   MONGO_DB_NAME=cattle_estrus
   REDIS_URL=redis://localhost:6379
   SECRET_KEY=your-secret-key-here
   ADMIN_USERNAME=admin
   ADMIN_PASSWORD=your-secure-password
   ```

5. **Start MongoDB and Redis**
   ```bash
   # MongoDB
   mongod --dbpath /path/to/data
   
   # Redis
   redis-server
   ```

6. **Create admin user**
   ```bash
   python create_admin.py
   ```

7. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

8. **Access the dashboard**
   Open your browser and navigate to: `http://localhost:8000`

## 📊 Data Processing Pipeline

### 1. Sensor Data Collection
The system expects CSV data with the following columns:
- `time`: Timestamp
- `ax`, `ay`, `az`: Accelerometer (x, y, z axes)
- `gx`, `gy`, `gz`: Gyroscope (x, y, z axes)
- `mx`, `my`, `mz`: Magnetometer (x, y, z axes)
- `amb`: Ambient temperature
- `obj`: Object temperature

### 2. Feature Engineering
- **Magnitude Calculation**: Computes vector magnitudes for each sensor
- **Movement Score**: Combined metric from all sensors
- **Dynamic Thresholds**: Rolling quantiles for adaptive classification
- **Daily Aggregation**: Mean, standard deviation, and activity fractions

### 3. Activity Classification
```
movement_score < low_threshold  → Resting
low_threshold ≤ score < mid     → Eating
mid ≤ score < high_threshold    → Ruminating
score ≥ high_threshold          → Walking
```

### 4. Estrus Detection Algorithm
Multi-condition validation:
1. **Relative Check**: Activity score > baseline_mean + 1.5 × baseline_std
2. **Absolute Check**: Activity score > 1.25 × baseline_mean
3. **Walking Check**: Walking fraction > 30%
4. **Dominance Check**: Top activity day dominance ratio > 1.15

All conditions must be satisfied for estrus confirmation.

## 🔧 Configuration

### Analysis Parameters
Edit `script.py` to customize detection parameters:

```python
ROLLING_ACTIVITY_DAYS = 3           # Window for activity classification
ESTRUS_BASELINE_DAYS = 7            # Baseline window for estrus detection
RELATIVE_STD_MULTIPLIER = 1.5       # Relative threshold multiplier
ABSOLUTE_ACTIVITY_MULTIPLIER = 1.25 # Absolute threshold multiplier
WALKING_THRESHOLD = 0.30            # Minimum walking fraction
DOMINANCE_RATIO_THRESHOLD = 1.15    # Dominance validation threshold
```

## 🛠️ Project Structure

```
cattle_estrus_test/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic models
│   ├── database.py          # MongoDB connection
│   ├── user_db.py           # User database handler
│   ├── auth.py              # Authentication logic
│   ├── cache.py             # Redis caching
│   ├── analysis.py          # Estrus detection algorithms
│   ├── websocket.py         # WebSocket handlers
│   └── templates/
│       ├── dashboard.html   # Main dashboard
│       └── login.html       # Login page
├── script.py                # Standalone analysis script
├── precompute_daily_metrics.py  # Batch processing
├── create_admin.py          # Admin user creation
├── test_mongo.py            # MongoDB tests
├── test_mongo2.py           # Additional tests
├── test_perf.py             # Performance tests
├── debug_millis.py          # Debugging utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📈 Usage Examples

### Running Batch Analysis
Process a CSV file with sensor data:
```bash
python script.py
```

### Precomputing Daily Metrics
For large datasets, precompute metrics for faster queries:
```bash
python precompute_daily_metrics.py
```

### API Endpoints
- `POST /api/auth/login` - User authentication
- `GET /api/cattle` - List all cattle
- `POST /api/cattle` - Add new cattle
- `GET /api/cattle/{cattle_id}/data` - Get sensor data
- `POST /api/data/upload` - Upload sensor data
- `GET /api/analysis/daily/{cattle_id}` - Daily activity analysis
- `GET /api/estrus/detection/{cattle_id}` - Estrus detection results
- `WS /ws/{cattle_id}` - WebSocket for real-time updates

### WebSocket Example
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/cattle_123');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

## 🧪 Testing

Run the test suite:
```bash
# MongoDB connection tests
python test_mongo.py

# Performance tests
python test_perf.py
```

## 🔒 Security

- JWT-based authentication with token expiration
- Password hashing using bcrypt
- Role-based access control (Admin/User)
- Secure session management
- Input validation on all endpoints

## 📊 Algorithm Details

### PCA Activity Score
The system uses Principal Component Analysis to reduce sensor features into a single activity score:
1. Standardizes features (acc_mean, acc_std, gyro_mean, gyro_std, mag_mean, mag_std)
2. Applies PCA to extract first principal component
3. Uses this score for estrus detection

### Baseline Calculation
- Rolling 7-day window (configurable)
- Shifted by 1 day to prevent self-comparison
- Mean and standard deviation computed for thresholds

## 🚧 Troubleshooting

### Common Issues

**MongoDB Connection Failed**
- Ensure MongoDB is running: `mongod --version`
- Check connection string in `.env`

**Redis Connection Error**
- Start Redis: `redis-server`
- Verify Redis is listening on port 6379

**No Estrus Detected**
- Check if data spans multiple days
- Verify sensor data quality
- Adjust detection thresholds if needed

**Import Errors**
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

## 📝 Future Enhancements

- [ ] Mobile application for field monitoring
- [ ] SMS/Email alerts for estrus detection
- [ ] Multi-cattle comparison dashboard
- [ ] Machine learning model training interface
- [ ] Export reports in PDF format
- [ ] Integration with farm management systems
- [ ] Predictive analytics for breeding optimization

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Peyal** - Initial work - [peyal939](https://github.com/peyal939)

## 🙏 Acknowledgments

- scikit-learn for machine learning algorithms
- FastAPI for the excellent web framework
- MongoDB and Redis for robust data storage
- All contributors and testers

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for improving cattle breeding efficiency**
