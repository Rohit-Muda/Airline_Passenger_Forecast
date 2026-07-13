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
- `app.py`
- `requirements.txt`
- `data/airline_passengers.csv`
- `models/lstm_model.keras`
- `models/scaler.pkl`

## Render Deployment
If you want to deploy on Render, use the included `render.yaml` file. Render will install dependencies from `requirements.txt` and run:

```bash
streamlit run app.py --server.port $PORT
```

## Vercel Deployment
If you want to deploy on Vercel, use the included `vercel.json` file. Vercel will use the Python builder for `app.py` and serve the Streamlit app as a Python endpoint.

## Docker Deployment
A `Dockerfile` is included for container deployment. Build and run locally with:

```bash
docker build -t airline-passenger-forecast .
docker run -p 8080:8080 airline-passenger-forecast
```

## Notes
- The app uses the existing forecasting pipeline from the project source code.
- If the hosted environment cannot import TensorFlow-backed modules, the app will still start and show an informative fallback message.
- Keep `models/lstm_model.keras` and `models/scaler.pkl` in the repo for prediction support.
