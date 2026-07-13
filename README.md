# Airline Passenger Forecasting

This project builds a simple Streamlit app for forecasting future airline passenger traffic using an existing recurrent neural network (RNN) workflow.

## Features
- Upload or use the bundled airline passenger dataset
- Preview the dataset and key summary metrics
- Choose a forecast horizon in months
- Generate and view future passenger predictions
- Display the forecast in a table and interactive chart

## Project Structure
- app.py: Streamlit application entry point
- data/: contains the airline passenger dataset
- src/: forecasting and preprocessing modules
- models/: trained model and scaler files
- outputs/: generated plots or result artifacts

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Streamlit Deployment
This app is ready for Streamlit Cloud or similar hosting platforms.

Make sure the repository contains:
- app.py
- requirements.txt
- data/airline_passengers.csv
- models/lstm_model.keras
- models/scaler.pkl

## Notes
- The app uses the existing forecasting pipeline from the project source code.
- For best deployment results, install the dependencies listed in requirements.txt before launching the app.
