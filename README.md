# ✈️ Flight Prediction Analysis (Shiny)

Interactive **Shiny for Python** dashboard for exploring US flight delays and cancellations (2015), plus a machine learning model that predicts whether a flight will arrive late.

🔗 **Live demo:** https://buba-krojek.shinyapps.io/flights-analysis1/

## Dashboard

Six tabs (`dashboard/app.py`):

- **Dataset summary** - number of flights, average delay, % of cancellations
- **Airports** - traffic map, delay distribution and delay causes per airport
- **Lines** - compare delays across airlines
- **Seasonality** - delays and cancellations by month and day of week
- **Routes** - flight-connection map and median delays of top routes
- **Flight summary** - visualisation of a single flight

## Machine learning

Notebook: `machine_learning/prototyping.ipynb`

- **Task:** binary classification - is the arrival delay greater than 15 minutes (FAA standard)? About 18% of flights are delayed, so classes are imbalanced.
- **Data:** ~5.2M completed flights from 2015; extreme outliers (above the 99.9th percentile) and data-leakage columns removed.
- **Features:** calendar/time features, delay of the aircraft's previous flight, airport congestion and historical delay statistics per airport, airline, route and hour. 54 of 77 candidate features kept after LightGBM importance selection.
- **Validation:** time-based split - train on January-October, test on November-December.

| Model | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.757 | 0.472 | 0.389 | 0.600 |
| Random Forest | 0.782 | 0.509 | 0.461 | 0.568 |
| **LightGBM** | 0.782 | 0.514 | 0.492 | 0.539 |

The final LightGBM model, tuned with Optuna and trained on the full training set, reaches **AUC 0.785** and **F1 0.515** on the test set.

## Tech stack

Shiny for Python, Polars, Plotly, LightGBM, scikit-learn, Optuna, deployed on shinyapps.io.

## Data

Public Kaggle datasets: [`usdot/flight-delays`](https://www.kaggle.com/datasets/usdot/flight-delays) and [`yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018`](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018).

## Run locally

```bash
git clone https://github.com/bubakrojek/flight-prediction-analysis-shiny.git
cd flight-prediction-analysis-shiny
pip install -r dashboard/requirements.txt
shiny run --reload dashboard/app.py
```

The app will be available at http://127.0.0.1:8000.
