# Smart Irrigation Prediction

This project predicts whether irrigation is required using a Long Short-Term Memory (LSTM) model trained on historical crop, temperature, humidity, and soil data.

## Project Structure

Smart_Irrigation_LSTM/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── raw_dataset.xlsx
├── models/
│   ├── irrigation_lstm.keras
│   ├── scaler.pkl
│   └── threshold.pkl
├── notebook/
│   └── smart_irrigation_experiment.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── sequence_creation.py
│   ├── model_training.py
│   └── model_evaluation.py
├── template/
│   └── index.html
├── static/
│   └── style.css
└── .gitignore

## Setup

1. Create a virtual environment
2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   python app.py

## How it works

- The app accepts 20 historical observations.
- Each row must contain:
  Crop Type, Temperature, Humidity, Soil Moisture, Soil Temperature
- The data is transformed using the saved scaler.
- The LSTM model predicts irrigation requirement probability.
- The saved threshold decides whether irrigation is required or not.

## Notes

The project currently uses:
- `template/` (not `templates/`)
- `app.py` to serve the Flask UI