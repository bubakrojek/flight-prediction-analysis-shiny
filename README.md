# ✈️ Flight Prediction Analysis (Shiny)

Interactive **Shiny for Python** dashboard for exploring US flight delays and cancellations (2015), supported by data-preprocessing notebooks and machine learning prototypes for delay prediction.

🔗 **Live demo:** https://buba-krojek.shinyapps.io/flights-analysis1/

## Features

The dashboard (`dashboard/app.py`) is split into six tabs:

| Tab | What you can do |
|---|---|
| **Dataset summary** | Number of flights, average delay, % of cancelled flights, preview of the data |
| **Airports** | Map of outgoing flights filtered by state and minimum traffic; delay distribution (violin plot), median delay and delay-cause breakdown for a chosen airport |
| **Lines** | Compare average arrival delay across airlines within a chosen delay range; delay distribution and median for a single airline |
| **Seasonality** | Radial chart by month and day-of-week/month heatmap for delays or cancellations, with max / mean / min value boxes |
| **Routes** | Flight-connection map (line thickness = traffic) and median departure/arrival delay of top airport pairs |
| **Flight summary** | Visualisation of a single flight by flight number, date and airline |

## Machine learning

The `machine_learning/` folder contains notebooks for predicting whether a flight will arrive late (`prototyping.ipynb`; earlier experiments in `prototyping_old.ipynb`).

**Problem definition**

- Binary classification: `IS_DELAYED = 1` when `ARRIVAL_DELAY` > 15 minutes (the FAA standard for a delayed flight).
- Only completed flights are used (cancelled and diverted flights are excluded) - about 5.7 million flights from 2015.
- Class imbalance: ~17.8% of flights are delayed vs ~82.2% on time.

**Data preparation**

- No missing values in the selected subset.
- Extreme outliers removed: arrival delays above the 99.9th percentile (368 min, ~0.1% of rows), which are rare and practically unpredictable.
- Data-leakage prevention: columns known only after take-off or landing (`DEPARTURE_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `AIR_TIME`, `ELAPSED_TIME`, `WHEELS_ON`, `TAXI_IN`, `ARRIVAL_TIME`) and delay-cause columns (`AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`) are dropped.
- High-cardinality `FLIGHT_NUMBER` is dropped; `TAIL_NUMBER`, `DEPARTURE_DELAY`, `SCHEDULED_DEPARTURE` and `SCHEDULED_ARRIVAL` are used only for feature engineering.

**Input features**

Month, day, day of week, airline, origin and destination airport, scheduled departure and arrival time, scheduled flight time and distance, plus engineered features.

**Status:** model training and evaluation are work in progress - results will be added here.

## Tech stack

- **App:** Shiny for Python, shinywidgets
- **Data processing:** Polars (lazy `scan_parquet`), pandas, PyArrow
- **Visualisation:** Plotly, matplotlib
- **Data download:** kagglehub
- **Deployment:** rsconnect-python, shinyapps.io

## Data

The project uses public Kaggle datasets:

- [`usdot/flight-delays`](https://www.kaggle.com/datasets/usdot/flight-delays) - flights, airlines and airports (2015)
- [`yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018`](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/bubakrojek/flight-prediction-analysis-shiny.git
cd flight-prediction-analysis-shiny

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r dashboard/requirements.txt

# 4. (Optional) download raw data from Kaggle
pip install kagglehub
python main.py

# 5. Run the dashboard
shiny run --reload dashboard/app.py
```

The app will be available at http://127.0.0.1:8000.

## Deployment

The dashboard is deployed on shinyapps.io and available online at https://buba-krojek.shinyapps.io/flights-analysis1/.

To redeploy:

```bash
rsconnect deploy shiny dashboard/ --name <account> --title flights-analysis1
```

## License

