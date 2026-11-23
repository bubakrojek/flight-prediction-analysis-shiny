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

## Tech stack

- **App:** Shiny for Python, shinywidgets
- **Data processing:** Polars (lazy `scan_parquet`), pandas, PyArrow
- **Visualisation:** Plotly, matplotlib
- **Data download:** kagglehub
- **Deployment:** rsconnect-python, shinyapps.io

## Project structure

```
.
├── analysis/
│   ├── preprocessing.ipynb      # data cleaning, conversion to parquet
│   └── visualisation.ipynb      # exploratory visualisations
├── machine_learning/
│   ├── prototyping.ipynb        # delay-prediction models
│   └── prototyping_old.ipynb    # earlier experiments
├── dashboard/
│   ├── app.py                   # Shiny app (UI + server)
│   ├── utils/                   # data preprocessing and plotting helpers
│   ├── data/                    # parquet files
│   └── requirements.txt
├── main.py                      # downloads datasets from Kaggle
└── README.md
```

## Data

The project uses public Kaggle datasets:

- [`usdot/flight-delays`](https://www.kaggle.com/datasets/usdot/flight-delays) - flights, airlines and airports (2015)
- [`yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018`](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)

The app expects the following files in `dashboard/data/`:

```
flights.parquet
airlines.parquet
airports.parquet
```

Run `analysis/preprocessing.ipynb` to generate them from the raw CSV files.

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

## Machine learning

Prototype models for predicting flight delays live in `machine_learning/prototyping.ipynb`.

## License

TODO: add a license.
