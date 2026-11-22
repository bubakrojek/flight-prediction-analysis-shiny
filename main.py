import os

import kagglehub
from kagglehub import KaggleDatasetAdapter



def download_smaller():
  file_path = "flights.csv"
  df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "usdot/flight-delays",
    file_path,
    pandas_kwargs={"encoding": "ISO-8859-1",
                   "low_memory":False}



  )

  print(df.columns)

def download_bigger():
  file_path = "2015.csv"

  # Load the latest version
  df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018",
    file_path,
    pandas_kwargs={"encoding":"ISO-8859-1",
        "low_memory":False,
        "compression":'zip'
                   }

  )

  print(df.columns)


download_bigger()
download_smaller()