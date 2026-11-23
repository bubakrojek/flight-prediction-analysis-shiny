# Flight Prediction Analysis (Shiny)

Interactive Shiny for Python dashboard for exploring flight delays, backed by data preprocessing, visualisation notebooks and machine learning prototypes for predicting delays.

## Project structure

- `analysis/` - data preprocessing and visualisation notebooks (`preprocessing.ipynb`, `visualisation.ipynb`)
- `machine_learning/` - model prototyping notebooks (`prototyping.ipynb`)
- `dashboard/` - Shiny app (`app.py`, `utils/`) and its dependencies (`requirements.txt`)
- `main.py` - project entry point

## Run the dashboard locally

```bash
pip install -r dashboard/requirements.txt
shiny run dashboard/app.py
```
