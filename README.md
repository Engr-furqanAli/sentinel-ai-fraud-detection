
```markdown
# 🛡 Sentinel AI - Fraud Detection Platform

A production-ready fraud detection system that uses machine learning to identify fraudulent transactions in real-time. Built with FastAPI, scikit-learn, and Streamlit.

## ✨ Features

- **Real-time Fraud Detection** - Instant transaction analysis with risk scoring
- **Multi-Model Support** - Separate models for credit card and retail fraud
- **Batch Processing** - Upload CSV files for bulk transaction scanning
- **Custom Training** - Train models on your own datasets
- **Risk Classification** - Low/Medium/High risk levels with clear decisions
- **REST API** - Well-documented API endpoints for integration

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│    ML       │
│  Frontend   │◀────│   Backend    │◀────│   Models    │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.10 |
| ML Models | scikit-learn (RandomForest, XGBoost) |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Serialization | Joblib |
| API Documentation | Swagger UI (built-in) |

## 📁 Project Structure

```
sentinel-fraud-platform/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Business logic
│   │   ├── models/       # ML model wrappers
│   │   └── schemas/      # Pydantic models
│   └── main.py           # FastAPI entry point
│
├── frontend/
│   └── app.py            # Streamlit dashboard
│
├── ml_training/
│   ├── train.py               # Credit card model
│   └── train_retail_model.py  # Retail model
│
├── data/
│   ├── raw/             # Training datasets
│   └── models/          # Trained model files
│
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Engr-furqanAli/sentinel-fraud-platform.git
cd sentinel-fraud-platform
```

2. **Create virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the models**
```bash
# Train credit card fraud model
cd backend/ml_training
python train.py

# Train retail fraud model
python train_retail_model.py
```

5. **Start the backend server**
```bash
cd ../..  # Back to root
uvicorn backend.main:app --reload
```

6. **Start the frontend dashboard**
```bash
# In a new terminal
streamlit run frontend/app.py
```

The application will be available at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Credit card fraud prediction |
| POST | `/api/v1/predict_retail` | Retail fraud prediction |
| GET | `/api/v1/model_info` | Get model information |
| POST | `/api/v1/reload_model` | Reload models |

### Sample Request
```json
{
  "amount": 1500.50,
  "location": "New York",
  "device": "Mobile"
}
```

### Sample Response
```json
{
  "fraud_probability": 0.12,
  "risk_score": 12.0,
  "risk_level": "Low",
  "decision": "Approve"
}
```

## 🧠 ML Models

### Credit Card Fraud Model
- **Algorithm**: XGBoost Classifier
- **Features**: Amount, location, device, transaction patterns
- **Performance**: 95%+ accuracy on test data

### Retail Fraud Model
- **Algorithm**: Random Forest Classifier
- **Features**: Transaction amount, merchant category, location, device, time patterns, velocity scores
- **Performance**: 92%+ accuracy on test data

## 📈 Risk Classification

| Probability | Risk Level | Decision |
|-------------|------------|----------|
| < 15% | Low | Approve |
| 15% - 35% | Medium | Manual Review |
| > 35% | High | Block Transaction |

## 🔧 Configuration

Key configuration files:
- `backend/app/core/config.py` - API settings
- `ml_training/train.py` - Model training parameters
- `frontend/app.py` - Dashboard settings

## 🧪 Testing

```bash
# Run backend tests
pytest backend/tests/

# Test API endpoints
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"amount": 1000, "location": "NYC", "device": "Mobile"}'
```

## 📦 Dependencies

Main packages:
```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.0
xgboost==2.0.0
pandas==2.0.3
numpy==1.24.3
streamlit==1.28.1
joblib==1.3.2
pydantic==2.4.2
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Furqan Ali**
- GitHub: [@Engr-furqanAli](https://github.com/Engr-furqanAli)
- LinkedIn: [Furqan Ali](https://linkedin.com/in/furqan-ali-595552233)
- Portfolio: [furqan-ali.dev](https://furqan-ali.dev)

## 🙏 Acknowledgments

- Dataset sources: Credit Card Fraud Detection (Kaggle)
- Inspired by real-world fraud detection systems
- Built as a production-ready ML portfolio project

---

<p align="center">
  Made with ❤️ by Furqan Ali
</p>
```

