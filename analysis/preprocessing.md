```python
import os

import kagglehub
import pandas as pd
import polars as pl

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)

from kagglehub import KaggleDatasetAdapter
```

# Downloading dataset


```python
def download_data_to_pandas():
    path = kagglehub.dataset_download("usdot/flight-delays")
    flights_path = "flights.csv"
    airlines_path = "airlines.csv"
    airports_path = "airports.csv"
    print("Reading dataset: ", path)
    flights_pd = pd.read_csv(
        os.path.join(path, flights_path),
        encoding="ISO-8859-1",
        low_memory=False,
        dtype={
            'YEAR': 'category',
            'DAY': 'category',
            'MONTH': 'category',
            'DAY_OF_WEEK': 'category',
            'AIRLINE': 'category',
            'FLIGHT_NUMBER': 'category',
            'TAIL_NUMBER': 'category',
            'ORIGIN_AIRPORT': 'category',
            'DESTINATION_AIRPORT': 'category',

            'SCHEDULED_DEPARTURE': 'str',
            'DEPARTURE_TIME': 'str',

            'WHEELS_OFF': 'str',
            'WHEELS_ON': 'str',

            'SCHEDULED_ARRIVAL': 'str',
            'ARRIVAL_TIME': 'str',

            'CANCELLATION_REASON': 'category',
        }
    )
    airlines_pd = pd.read_csv(
        os.path.join(path, airlines_path),
        encoding="ISO-8859-1",
        low_memory=False,
        dtype={
            'IATA_CODE': 'category',
            'AIRLINE': 'category',
        }
    )
    airports_pd = pd.read_csv(
        os.path.join(path, airports_path),
        encoding="ISO-8859-1",
        low_memory=False,
        dtype={
            'IATA_CODE': 'category',
            'AIRPORT': 'category',
            'CITY': 'category',
            'STATE': 'category',
            'COUNTRY': 'category',

        }
    )
    return flights_pd, airlines_pd, airports_pd

def download_data_to_polars():
    path = kagglehub.dataset_download("usdot/flight-delays")
    flights_path = "flights.csv"
    airlines_path = "airlines.csv"
    airports_path = "airports.csv"
    print("Reading dataset: ", path)

    flights_pl = pl.read_csv(
        os.path.join(path, flights_path),
        low_memory=False,
        encoding="ISO-8859-1",
        schema_overrides = {
            'YEAR': pl.Categorical,
            'DAY': pl.Categorical,
            'MONTH': pl.Categorical,
            'DAY_OF_WEEK': pl.Categorical,
            'AIRLINE': pl.Categorical,
            'FLIGHT_NUMBER': pl.Categorical,
            'TAIL_NUMBER': pl.Categorical,
            'ORIGIN_AIRPORT': pl.Categorical,
            'DESTINATION_AIRPORT': pl.Categorical,

            'SCHEDULED_DEPARTURE': pl.String,
            'DEPARTURE_TIME': pl.String,

            'WHEELS_OFF': pl.String,
            'WHEELS_ON': pl.String,

            'SCHEDULED_ARRIVAL': pl.String,
            'ARRIVAL_TIME': pl.String,

            'CANCELLATION_REASON': pl.Categorical,
        }
    )
        #TODO - add creating dataframes for other datasets


def download_bigger():
    file_path = "2015.csv"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018",
        file_path,
        pandas_kwargs={"encoding": "ISO-8859-1",
                       "low_memory": False,
                       "compression": 'zip'
                       }

    )
    return df
```


```python
flights, airlines, airports = download_data_to_pandas()
```

    Reading dataset:  C:\Users\kb412\.cache\kagglehub\datasets\usdot\flight-delays\versions\1
    

# Airports


```python
airports.head()
```


```python
print("No. nulls in airports:\n", airports.isna().sum())
```


```python
airports.loc[airports.isna().any(axis=1)]
```

There are 3 empty coordinates in the airports data. This kind of nulls can be inserted manually, after checking the coordinates.


```python
airports.loc[airports['IATA_CODE'] == "ECP", ["LATITUDE", 'LONGITUDE']] = [30.35824, -85.79560]
airports.loc[airports['IATA_CODE'] == "PBG", ["LATITUDE", 'LONGITUDE']] = [44.65094, -73.46814]
airports.loc[airports['IATA_CODE'] == "UST", ["LATITUDE", 'LONGITUDE']] = [29.95925, -81.33973]
print("No. nulls in airports:\n", airports.isna().any().sum())
```

# Airlines


```python
airlines.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IATA_CODE</th>
      <th>AIRLINE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UA</td>
      <td>United Air Lines Inc.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AA</td>
      <td>American Airlines Inc.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>US Airways Inc.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F9</td>
      <td>Frontier Airlines Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B6</td>
      <td>JetBlue Airways</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("No. nulls in airlines:\n", airlines.isna().sum())
```

    No. nulls in airlines:
     IATA_CODE    0
    AIRLINE      0
    dtype: int64
    

Luckily, there are no empty rows in the airlines dataset

# Flights


```python
import polars as pl

flights = pl.from_pandas(flights).lazy()
airlines = pl.from_pandas(airlines).lazy()
airports = pl.from_pandas(airports).lazy()
```

## Initial insight into the data


```python
flights.head(10).collect()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 31)</small><table border="1" class="dataframe"><thead><tr><th>YEAR</th><th>MONTH</th><th>DAY</th><th>DAY_OF_WEEK</th><th>AIRLINE</th><th>FLIGHT_NUMBER</th><th>TAIL_NUMBER</th><th>ORIGIN_AIRPORT</th><th>DESTINATION_AIRPORT</th><th>SCHEDULED_DEPARTURE</th><th>DEPARTURE_TIME</th><th>DEPARTURE_DELAY</th><th>TAXI_OUT</th><th>WHEELS_OFF</th><th>SCHEDULED_TIME</th><th>ELAPSED_TIME</th><th>AIR_TIME</th><th>DISTANCE</th><th>WHEELS_ON</th><th>TAXI_IN</th><th>SCHEDULED_ARRIVAL</th><th>ARRIVAL_TIME</th><th>ARRIVAL_DELAY</th><th>DIVERTED</th><th>CANCELLED</th><th>CANCELLATION_REASON</th><th>AIR_SYSTEM_DELAY</th><th>SECURITY_DELAY</th><th>AIRLINE_DELAY</th><th>LATE_AIRCRAFT_DELAY</th><th>WEATHER_DELAY</th></tr><tr><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>str</td><td>str</td><td>f64</td><td>f64</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td><td>str</td><td>f64</td><td>str</td><td>str</td><td>f64</td><td>i64</td><td>i64</td><td>cat</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;AS&quot;</td><td>&quot;98&quot;</td><td>&quot;N407AS&quot;</td><td>&quot;ANC&quot;</td><td>&quot;SEA&quot;</td><td>&quot;0005&quot;</td><td>&quot;2354&quot;</td><td>-11.0</td><td>21.0</td><td>&quot;0015&quot;</td><td>205.0</td><td>194.0</td><td>169.0</td><td>1448</td><td>&quot;0404&quot;</td><td>4.0</td><td>&quot;0430&quot;</td><td>&quot;0408&quot;</td><td>-22.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;AA&quot;</td><td>&quot;2336&quot;</td><td>&quot;N3KUAA&quot;</td><td>&quot;LAX&quot;</td><td>&quot;PBI&quot;</td><td>&quot;0010&quot;</td><td>&quot;0002&quot;</td><td>-8.0</td><td>12.0</td><td>&quot;0014&quot;</td><td>280.0</td><td>279.0</td><td>263.0</td><td>2330</td><td>&quot;0737&quot;</td><td>4.0</td><td>&quot;0750&quot;</td><td>&quot;0741&quot;</td><td>-9.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;US&quot;</td><td>&quot;840&quot;</td><td>&quot;N171US&quot;</td><td>&quot;SFO&quot;</td><td>&quot;CLT&quot;</td><td>&quot;0020&quot;</td><td>&quot;0018&quot;</td><td>-2.0</td><td>16.0</td><td>&quot;0034&quot;</td><td>286.0</td><td>293.0</td><td>266.0</td><td>2296</td><td>&quot;0800&quot;</td><td>11.0</td><td>&quot;0806&quot;</td><td>&quot;0811&quot;</td><td>5.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;AA&quot;</td><td>&quot;258&quot;</td><td>&quot;N3HYAA&quot;</td><td>&quot;LAX&quot;</td><td>&quot;MIA&quot;</td><td>&quot;0020&quot;</td><td>&quot;0015&quot;</td><td>-5.0</td><td>15.0</td><td>&quot;0030&quot;</td><td>285.0</td><td>281.0</td><td>258.0</td><td>2342</td><td>&quot;0748&quot;</td><td>8.0</td><td>&quot;0805&quot;</td><td>&quot;0756&quot;</td><td>-9.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;AS&quot;</td><td>&quot;135&quot;</td><td>&quot;N527AS&quot;</td><td>&quot;SEA&quot;</td><td>&quot;ANC&quot;</td><td>&quot;0025&quot;</td><td>&quot;0024&quot;</td><td>-1.0</td><td>11.0</td><td>&quot;0035&quot;</td><td>235.0</td><td>215.0</td><td>199.0</td><td>1448</td><td>&quot;0254&quot;</td><td>5.0</td><td>&quot;0320&quot;</td><td>&quot;0259&quot;</td><td>-21.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;DL&quot;</td><td>&quot;806&quot;</td><td>&quot;N3730B&quot;</td><td>&quot;SFO&quot;</td><td>&quot;MSP&quot;</td><td>&quot;0025&quot;</td><td>&quot;0020&quot;</td><td>-5.0</td><td>18.0</td><td>&quot;0038&quot;</td><td>217.0</td><td>230.0</td><td>206.0</td><td>1589</td><td>&quot;0604&quot;</td><td>6.0</td><td>&quot;0602&quot;</td><td>&quot;0610&quot;</td><td>8.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;NK&quot;</td><td>&quot;612&quot;</td><td>&quot;N635NK&quot;</td><td>&quot;LAS&quot;</td><td>&quot;MSP&quot;</td><td>&quot;0025&quot;</td><td>&quot;0019&quot;</td><td>-6.0</td><td>11.0</td><td>&quot;0030&quot;</td><td>181.0</td><td>170.0</td><td>154.0</td><td>1299</td><td>&quot;0504&quot;</td><td>5.0</td><td>&quot;0526&quot;</td><td>&quot;0509&quot;</td><td>-17.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;US&quot;</td><td>&quot;2013&quot;</td><td>&quot;N584UW&quot;</td><td>&quot;LAX&quot;</td><td>&quot;CLT&quot;</td><td>&quot;0030&quot;</td><td>&quot;0044&quot;</td><td>14.0</td><td>13.0</td><td>&quot;0057&quot;</td><td>273.0</td><td>249.0</td><td>228.0</td><td>2125</td><td>&quot;0745&quot;</td><td>8.0</td><td>&quot;0803&quot;</td><td>&quot;0753&quot;</td><td>-10.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;AA&quot;</td><td>&quot;1112&quot;</td><td>&quot;N3LAAA&quot;</td><td>&quot;SFO&quot;</td><td>&quot;DFW&quot;</td><td>&quot;0030&quot;</td><td>&quot;0019&quot;</td><td>-11.0</td><td>17.0</td><td>&quot;0036&quot;</td><td>195.0</td><td>193.0</td><td>173.0</td><td>1464</td><td>&quot;0529&quot;</td><td>3.0</td><td>&quot;0545&quot;</td><td>&quot;0532&quot;</td><td>-13.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;DL&quot;</td><td>&quot;1173&quot;</td><td>&quot;N826DN&quot;</td><td>&quot;LAS&quot;</td><td>&quot;ATL&quot;</td><td>&quot;0030&quot;</td><td>&quot;0033&quot;</td><td>3.0</td><td>12.0</td><td>&quot;0045&quot;</td><td>221.0</td><td>203.0</td><td>186.0</td><td>1747</td><td>&quot;0651&quot;</td><td>5.0</td><td>&quot;0711&quot;</td><td>&quot;0656&quot;</td><td>-15.0</td><td>0</td><td>0</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td></tr></tbody></table></div>




```python
flights.collect_schema().names()
```




    ['YEAR',
     'MONTH',
     'DAY',
     'DAY_OF_WEEK',
     'AIRLINE',
     'FLIGHT_NUMBER',
     'TAIL_NUMBER',
     'ORIGIN_AIRPORT',
     'DESTINATION_AIRPORT',
     'SCHEDULED_DEPARTURE',
     'DEPARTURE_TIME',
     'DEPARTURE_DELAY',
     'TAXI_OUT',
     'WHEELS_OFF',
     'SCHEDULED_TIME',
     'ELAPSED_TIME',
     'AIR_TIME',
     'DISTANCE',
     'WHEELS_ON',
     'TAXI_IN',
     'SCHEDULED_ARRIVAL',
     'ARRIVAL_TIME',
     'ARRIVAL_DELAY',
     'DIVERTED',
     'CANCELLED',
     'CANCELLATION_REASON',
     'AIR_SYSTEM_DELAY',
     'SECURITY_DELAY',
     'AIRLINE_DELAY',
     'LATE_AIRCRAFT_DELAY',
     'WEATHER_DELAY']




```python
(flights.describe())
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 32)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>YEAR</th><th>MONTH</th><th>DAY</th><th>DAY_OF_WEEK</th><th>AIRLINE</th><th>FLIGHT_NUMBER</th><th>TAIL_NUMBER</th><th>ORIGIN_AIRPORT</th><th>DESTINATION_AIRPORT</th><th>SCHEDULED_DEPARTURE</th><th>DEPARTURE_TIME</th><th>DEPARTURE_DELAY</th><th>TAXI_OUT</th><th>WHEELS_OFF</th><th>SCHEDULED_TIME</th><th>ELAPSED_TIME</th><th>AIR_TIME</th><th>DISTANCE</th><th>WHEELS_ON</th><th>TAXI_IN</th><th>SCHEDULED_ARRIVAL</th><th>ARRIVAL_TIME</th><th>ARRIVAL_DELAY</th><th>DIVERTED</th><th>CANCELLED</th><th>CANCELLATION_REASON</th><th>AIR_SYSTEM_DELAY</th><th>SECURITY_DELAY</th><th>AIRLINE_DELAY</th><th>LATE_AIRCRAFT_DELAY</th><th>WEATHER_DELAY</th></tr><tr><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>str</td><td>f64</td><td>f64</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>str</td><td>f64</td><td>str</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>str</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5804358&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5819079&quot;</td><td>&quot;5732926&quot;</td><td>5.732926e6</td><td>5.730032e6</td><td>&quot;5730032&quot;</td><td>5.819073e6</td><td>5.714008e6</td><td>5.714008e6</td><td>5.819079e6</td><td>&quot;5726566&quot;</td><td>5.726566e6</td><td>&quot;5819079&quot;</td><td>&quot;5726566&quot;</td><td>5.714008e6</td><td>5.819079e6</td><td>5.819079e6</td><td>&quot;89884&quot;</td><td>1.063439e6</td><td>1.063439e6</td><td>1.063439e6</td><td>1.063439e6</td><td>1.063439e6</td></tr><tr><td>&quot;null_count&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;14721&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;0&quot;</td><td>&quot;86153&quot;</td><td>86153.0</td><td>89047.0</td><td>&quot;89047&quot;</td><td>6.0</td><td>105071.0</td><td>105071.0</td><td>0.0</td><td>&quot;92513&quot;</td><td>92513.0</td><td>&quot;0&quot;</td><td>&quot;92513&quot;</td><td>105071.0</td><td>0.0</td><td>0.0</td><td>&quot;5729195&quot;</td><td>4.75564e6</td><td>4.75564e6</td><td>4.75564e6</td><td>4.75564e6</td><td>4.75564e6</td></tr><tr><td>&quot;mean&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>9.370158</td><td>16.071662</td><td>null</td><td>141.685892</td><td>137.006189</td><td>113.511628</td><td>822.356495</td><td>null</td><td>7.434971</td><td>null</td><td>null</td><td>4.407057</td><td>0.00261</td><td>0.015446</td><td>null</td><td>13.480568</td><td>0.076154</td><td>18.969547</td><td>23.472838</td><td>2.91529</td></tr><tr><td>&quot;std&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>37.080942</td><td>8.895574</td><td>null</td><td>75.210582</td><td>74.211072</td><td>72.230822</td><td>607.784287</td><td>null</td><td>5.638548</td><td>null</td><td>null</td><td>39.271297</td><td>0.05102</td><td>0.12332</td><td>null</td><td>28.003679</td><td>2.14346</td><td>48.161642</td><td>43.197018</td><td>20.433336</td></tr><tr><td>&quot;min&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>&quot;0001&quot;</td><td>&quot;0001&quot;</td><td>-82.0</td><td>1.0</td><td>&quot;0001&quot;</td><td>18.0</td><td>14.0</td><td>7.0</td><td>21.0</td><td>&quot;0001&quot;</td><td>1.0</td><td>&quot;0001&quot;</td><td>&quot;0001&quot;</td><td>-87.0</td><td>0.0</td><td>0.0</td><td>null</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;25%&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>-5.0</td><td>11.0</td><td>null</td><td>85.0</td><td>82.0</td><td>60.0</td><td>373.0</td><td>null</td><td>4.0</td><td>null</td><td>null</td><td>-13.0</td><td>0.0</td><td>0.0</td><td>null</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;50%&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>-2.0</td><td>14.0</td><td>null</td><td>123.0</td><td>118.0</td><td>94.0</td><td>647.0</td><td>null</td><td>6.0</td><td>null</td><td>null</td><td>-5.0</td><td>0.0</td><td>0.0</td><td>null</td><td>2.0</td><td>0.0</td><td>2.0</td><td>3.0</td><td>0.0</td></tr><tr><td>&quot;75%&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>7.0</td><td>19.0</td><td>null</td><td>173.0</td><td>168.0</td><td>144.0</td><td>1062.0</td><td>null</td><td>9.0</td><td>null</td><td>null</td><td>8.0</td><td>0.0</td><td>0.0</td><td>null</td><td>18.0</td><td>0.0</td><td>19.0</td><td>29.0</td><td>0.0</td></tr><tr><td>&quot;max&quot;</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>null</td><td>&quot;2359&quot;</td><td>&quot;2400&quot;</td><td>1988.0</td><td>225.0</td><td>&quot;2400&quot;</td><td>718.0</td><td>766.0</td><td>690.0</td><td>4983.0</td><td>&quot;2400&quot;</td><td>248.0</td><td>&quot;2400&quot;</td><td>&quot;2400&quot;</td><td>1971.0</td><td>1.0</td><td>1.0</td><td>null</td><td>1134.0</td><td>573.0</td><td>1971.0</td><td>1331.0</td><td>1211.0</td></tr></tbody></table></div>




```python
flights.collect_schema().dtypes()
```




    [Categorical,
     Categorical,
     Categorical,
     Categorical,
     Categorical,
     Categorical,
     Categorical,
     Categorical,
     Categorical,
     String,
     String,
     Float64,
     Float64,
     String,
     Float64,
     Float64,
     Float64,
     Int64,
     String,
     Float64,
     String,
     String,
     Float64,
     Int64,
     Int64,
     Categorical,
     Float64,
     Float64,
     Float64,
     Float64,
     Float64]



## Changing time columns to appropriate data type


```python
time_columns = [
    'SCHEDULED_DEPARTURE',
    'DEPARTURE_TIME',
    'WHEELS_OFF',
    'WHEELS_ON',
    'SCHEDULED_ARRIVAL',
    'ARRIVAL_TIME',
]
schema = flights.collect_schema()

for col in time_columns:
    if schema[col] == pl.String:
        flights = flights.with_columns(
            pl.col(col)
            .str.zfill(4)
            .str.replace("2400", "0000")
            .str.to_time("%H%M", strict=False).alias(col)
        )

```

## Dealing with nulls


```python
print("No. nulls in airlines:\n",
      flights
      .null_count()
      .unpivot(variable_name='tmp', value_name='null_count')
      .filter(pl.col('null_count') > 0)
      .collect())
```

    No. nulls in airlines:
     shape: (18, 2)
    ┌─────────────────────┬────────────┐
    │ tmp                 ┆ null_count │
    │ ---                 ┆ ---        │
    │ str                 ┆ u32        │
    ╞═════════════════════╪════════════╡
    │ TAIL_NUMBER         ┆ 14721      │
    │ DEPARTURE_TIME      ┆ 86153      │
    │ DEPARTURE_DELAY     ┆ 86153      │
    │ TAXI_OUT            ┆ 89047      │
    │ WHEELS_OFF          ┆ 89047      │
    │ SCHEDULED_TIME      ┆ 6          │
    │ ELAPSED_TIME        ┆ 105071     │
    │ AIR_TIME            ┆ 105071     │
    │ WHEELS_ON           ┆ 92513      │
    │ TAXI_IN             ┆ 92513      │
    │ ARRIVAL_TIME        ┆ 92513      │
    │ ARRIVAL_DELAY       ┆ 105071     │
    │ CANCELLATION_REASON ┆ 5729195    │
    │ AIR_SYSTEM_DELAY    ┆ 4755640    │
    │ SECURITY_DELAY      ┆ 4755640    │
    │ AIRLINE_DELAY       ┆ 4755640    │
    │ LATE_AIRCRAFT_DELAY ┆ 4755640    │
    │ WEATHER_DELAY       ┆ 4755640    │
    └─────────────────────┴────────────┘
    

### Tail number

We can try to start dealing with null on Tail Numbers.

There is  possibility that some Airline doesn't use many different Tail numbers.


```python
(flights.group_by('AIRLINE')
 .agg(pl.col('TAIL_NUMBER')
      .n_unique()
      .alias('unique_tail_numbers'))
 .sort('unique_tail_numbers')
 .collect())
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (14, 2)</small><table border="1" class="dataframe"><thead><tr><th>AIRLINE</th><th>unique_tail_numbers</th></tr><tr><td>cat</td><td>u32</td></tr></thead><tbody><tr><td>&quot;HA&quot;</td><td>51</td></tr><tr><td>&quot;VX&quot;</td><td>57</td></tr><tr><td>&quot;F9&quot;</td><td>63</td></tr><tr><td>&quot;NK&quot;</td><td>79</td></tr><tr><td>&quot;AS&quot;</td><td>147</td></tr><tr><td>&quot;MQ&quot;</td><td>204</td></tr><tr><td>&quot;B6&quot;</td><td>215</td></tr><tr><td>&quot;US&quot;</td><td>352</td></tr><tr><td>&quot;OO&quot;</td><td>384</td></tr><tr><td>&quot;EV&quot;</td><td>391</td></tr><tr><td>&quot;WN&quot;</td><td>705</td></tr><tr><td>&quot;UA&quot;</td><td>722</td></tr><tr><td>&quot;DL&quot;</td><td>829</td></tr><tr><td>&quot;AA&quot;</td><td>1045</td></tr></tbody></table></div>



Imputing with mode might be impossible for all. We can try to impute this type of data using knn. There is a possibility where there are e.g., many flights that are basically the same but on the other day.


```python
print(flights
      .filter(
    ~pl.col('TAIL_NUMBER').is_null(),
    pl.col('ORIGIN_AIRPORT') == 'ANC',
    pl.col('DESTINATION_AIRPORT') == 'DEN',
    pl.col('AIRLINE') == 'UA',
    pl.col('DAY_OF_WEEK') == '1',
    pl.col('FLIGHT_NUMBER') == '1104'
)
      .select(pl.col('TAIL_NUMBER')
              .value_counts())
      .unnest('TAIL_NUMBER')
      .sort('count', descending=True)
      .collect())

```

    shape: (44, 2)
    ┌─────────────┬───────┐
    │ TAIL_NUMBER ┆ count │
    │ ---         ┆ ---   │
    │ cat         ┆ u32   │
    ╞═════════════╪═══════╡
    │ N76519      ┆ 2     │
    │ N73270      ┆ 2     │
    │ N73299      ┆ 1     │
    │ N66828      ┆ 1     │
    │ N39297      ┆ 1     │
    │ N76254      ┆ 1     │
    │ N27205      ┆ 1     │
    │ N73256      ┆ 1     │
    │ N14250      ┆ 1     │
    │ N73275      ┆ 1     │
    │ N17245      ┆ 1     │
    │ N36247      ┆ 1     │
    │ N38417      ┆ 1     │
    │ N16234      ┆ 1     │
    │ N66825      ┆ 1     │
    │ N37267      ┆ 1     │
    │ N36207      ┆ 1     │
    │ N78285      ┆ 1     │
    │ N26210      ┆ 1     │
    │ N37427      ┆ 1     │
    │ N33292      ┆ 1     │
    │ N76269      ┆ 1     │
    │ N76528      ┆ 1     │
    │ N14231      ┆ 1     │
    │ N37273      ┆ 1     │
    │ N14214      ┆ 1     │
    │ N76522      ┆ 1     │
    │ N38424      ┆ 1     │
    │ N12225      ┆ 1     │
    │ N17233      ┆ 1     │
    │ N37437      ┆ 1     │
    │ N38473      ┆ 1     │
    │ N78509      ┆ 1     │
    │ N68811      ┆ 1     │
    │ N76508      ┆ 1     │
    │ N18223      ┆ 1     │
    │ N66841      ┆ 1     │
    │ N76502      ┆ 1     │
    │ N35204      ┆ 1     │
    │ N33286      ┆ 1     │
    │ N67812      ┆ 1     │
    │ N66848      ┆ 1     │
    │ N24202      ┆ 1     │
    │ N78524      ┆ 1     │
    └─────────────┴───────┘
    

It looks like that imputation might be challenging, there aren't any repetitive tail numbers that can be used for imputation.
Tail number might not be important for predicting, maybe only for visualization, so we can drop this column in a further part or drop rows when visualizing.

### Cancellation reason

The next column for nulls analysis might be Cancellation reason with 5729195 nulls. Reason why those nulls occur is that those flights weren't canceled, so there is no reason, we can create new class "None" which can represent those nulls.
The number of not null values can also tell us many useful information for further imputations.


```python
flights = flights.with_columns(
    pl.col('CANCELLATION_REASON').fill_null("None")
)
```

### Delays

There is very similar situation with delays, if there is a Null it means there was none delay, so we can impute this blanks with 0.


```python
flights = flights.with_columns(
    pl.col('AIR_SYSTEM_DELAY').fill_null(0),
    pl.col('SECURITY_DELAY').fill_null(0),
    pl.col('AIRLINE_DELAY').fill_null(0),
    pl.col('LATE_AIRCRAFT_DELAY').fill_null(0),
    pl.col('WEATHER_DELAY').fill_null(0)
)
```

### Diverted and canceled analysis


```python
(flights.select(
    ((pl.col('DIVERTED') == 1) | (pl.col('CANCELLED') == 1)).sum().alias('diverted_or_cancelled'),
    pl.col('DIVERTED').sum().alias('diverted'),
    pl.col('CANCELLED').sum().alias('cancelled')
).collect())
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (1, 3)</small><table border="1" class="dataframe"><thead><tr><th>diverted_or_cancelled</th><th>diverted</th><th>cancelled</th></tr><tr><td>u32</td><td>i64</td><td>i64</td></tr></thead><tbody><tr><td>105071</td><td>15187</td><td>89884</td></tr></tbody></table></div>



There are some columns that are empty because of flight being canceled or diverted, specifically:
- │ ELAPSED_TIME        ┆ 105071     │
- │ AIR_TIME            ┆ 105071     │
- │ ARRIVAL_DELAY       ┆ 105071     │

This is because  these columns aren't inserted automatically by system and there is no sense to calculate delay or time in the air if the plane flew to another airport or didn't left origin one. We won't impute them with anything because it won't have any sense, if we want to train model for e.g., delays we should drop rows diverted or canceled flights.

#### Diverted


```python
print("No. nulls in airlines:\n",
      flights
      .filter((pl.col('DIVERTED') == 1))
      .null_count()
      .unpivot(variable_name='tmp', value_name='null_count')
      .filter(pl.col('null_count') > 0)
      .collect())
```

    No. nulls in airlines:
     shape: (7, 2)
    ┌────────────────┬────────────┐
    │ tmp            ┆ null_count │
    │ ---            ┆ ---        │
    │ str            ┆ u32        │
    ╞════════════════╪════════════╡
    │ SCHEDULED_TIME ┆ 1          │
    │ ELAPSED_TIME   ┆ 15187      │
    │ AIR_TIME       ┆ 15187      │
    │ WHEELS_ON      ┆ 2629       │
    │ TAXI_IN        ┆ 2629       │
    │ ARRIVAL_TIME   ┆ 2629       │
    │ ARRIVAL_DELAY  ┆ 15187      │
    └────────────────┴────────────┘
    

There are 15187 diverted flights, and there is the same number of blanks in all columns apart from WHEELS_ON, TAXI_IN and ARRIVAL_TIME, as it was said before in other columns there is no sens to insert them.

The number of nulls in these 3 columns is showing us that most of the flights that were diverted finally reached the destination, but 2629 of them didn't and that's why columns aren't filled.

#### Cancelled



```python
print("No. nulls in airlines:\n",
      flights
      .filter((pl.col('CANCELLED') == 1))
      .null_count()
      .unpivot(variable_name='tmp', value_name='null_count')
      .filter(pl.col('null_count') > 0)
      .collect())
```

    No. nulls in airlines:
     shape: (12, 2)
    ┌─────────────────┬────────────┐
    │ tmp             ┆ null_count │
    │ ---             ┆ ---        │
    │ str             ┆ u32        │
    ╞═════════════════╪════════════╡
    │ TAIL_NUMBER     ┆ 14721      │
    │ DEPARTURE_TIME  ┆ 86153      │
    │ DEPARTURE_DELAY ┆ 86153      │
    │ TAXI_OUT        ┆ 89047      │
    │ WHEELS_OFF      ┆ 89047      │
    │ SCHEDULED_TIME  ┆ 5          │
    │ ELAPSED_TIME    ┆ 89884      │
    │ AIR_TIME        ┆ 89884      │
    │ WHEELS_ON       ┆ 89884      │
    │ TAXI_IN         ┆ 89884      │
    │ ARRIVAL_TIME    ┆ 89884      │
    │ ARRIVAL_DELAY   ┆ 89884      │
    └─────────────────┴────────────┘
    

As we can see there are many nulls at canceled flights.

Now we can analyze all of them except TAIL_NUMBER and SCHEDULED_TIME.


```python
q1 = flights.filter(
    (pl.col('CANCELLED') == 1),
    pl.col('DEPARTURE_TIME').is_null()
).head(1).collect()
q2 = flights.filter(
    (pl.col('CANCELLED') == 1),
    pl.col('TAXI_OUT').is_null(),
    ~pl.col('DEPARTURE_TIME').is_null()
).head(1).collect()
q3 = flights.filter(
    (pl.col('CANCELLED') == 1),
    pl.col('ELAPSED_TIME').is_null(),
    ~pl.col('WHEELS_OFF').is_null()
).head(1).collect()
pl.concat([q1, q2, q3])
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (3, 31)</small><table border="1" class="dataframe"><thead><tr><th>YEAR</th><th>MONTH</th><th>DAY</th><th>DAY_OF_WEEK</th><th>AIRLINE</th><th>FLIGHT_NUMBER</th><th>TAIL_NUMBER</th><th>ORIGIN_AIRPORT</th><th>DESTINATION_AIRPORT</th><th>SCHEDULED_DEPARTURE</th><th>DEPARTURE_TIME</th><th>DEPARTURE_DELAY</th><th>TAXI_OUT</th><th>WHEELS_OFF</th><th>SCHEDULED_TIME</th><th>ELAPSED_TIME</th><th>AIR_TIME</th><th>DISTANCE</th><th>WHEELS_ON</th><th>TAXI_IN</th><th>SCHEDULED_ARRIVAL</th><th>ARRIVAL_TIME</th><th>ARRIVAL_DELAY</th><th>DIVERTED</th><th>CANCELLED</th><th>CANCELLATION_REASON</th><th>AIR_SYSTEM_DELAY</th><th>SECURITY_DELAY</th><th>AIRLINE_DELAY</th><th>LATE_AIRCRAFT_DELAY</th><th>WEATHER_DELAY</th></tr><tr><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>time</td><td>time</td><td>f64</td><td>f64</td><td>time</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td><td>time</td><td>f64</td><td>time</td><td>time</td><td>f64</td><td>i64</td><td>i64</td><td>cat</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;AS&quot;</td><td>&quot;136&quot;</td><td>&quot;N431AS&quot;</td><td>&quot;ANC&quot;</td><td>&quot;SEA&quot;</td><td>01:35:00</td><td>null</td><td>null</td><td>null</td><td>null</td><td>205.0</td><td>null</td><td>null</td><td>1448</td><td>null</td><td>null</td><td>06:00:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;A&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;MQ&quot;</td><td>&quot;2758&quot;</td><td>&quot;N939MQ&quot;</td><td>&quot;ROW&quot;</td><td>&quot;DFW&quot;</td><td>11:10:00</td><td>12:29:00</td><td>79.0</td><td>null</td><td>null</td><td>85.0</td><td>null</td><td>null</td><td>435</td><td>null</td><td>null</td><td>13:35:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;B&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;1&quot;</td><td>&quot;1&quot;</td><td>&quot;4&quot;</td><td>&quot;EV&quot;</td><td>&quot;4503&quot;</td><td>&quot;N14514&quot;</td><td>&quot;IAH&quot;</td><td>&quot;BRO&quot;</td><td>11:34:00</td><td>11:29:00</td><td>-5.0</td><td>20.0</td><td>11:49:00</td><td>76.0</td><td>null</td><td>null</td><td>308</td><td>null</td><td>null</td><td>12:50:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;B&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></tbody></table></div>



As we can see, there are three types of nulls.

- Departure time is null, the plane didn't even leave the terminal, so other columns are also empty.
- Departure time is not null, but wheels off are empty, it means that the plane left the terminal but didn't take off.
- This time the plane took off, so the previous stats are populated, but it didn't land or destination airport nor was diverted,

So any of these null aren't unwanted and it's normal behavior.

There are also nulls in SCHEDULED_TIME which should be investigated. Looking at the amount of them they probably shouldn't exist. It is also occurring in one of the diverted flight.


```python
flights.filter(pl.col('SCHEDULED_TIME').is_null()).collect()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (6, 31)</small><table border="1" class="dataframe"><thead><tr><th>YEAR</th><th>MONTH</th><th>DAY</th><th>DAY_OF_WEEK</th><th>AIRLINE</th><th>FLIGHT_NUMBER</th><th>TAIL_NUMBER</th><th>ORIGIN_AIRPORT</th><th>DESTINATION_AIRPORT</th><th>SCHEDULED_DEPARTURE</th><th>DEPARTURE_TIME</th><th>DEPARTURE_DELAY</th><th>TAXI_OUT</th><th>WHEELS_OFF</th><th>SCHEDULED_TIME</th><th>ELAPSED_TIME</th><th>AIR_TIME</th><th>DISTANCE</th><th>WHEELS_ON</th><th>TAXI_IN</th><th>SCHEDULED_ARRIVAL</th><th>ARRIVAL_TIME</th><th>ARRIVAL_DELAY</th><th>DIVERTED</th><th>CANCELLED</th><th>CANCELLATION_REASON</th><th>AIR_SYSTEM_DELAY</th><th>SECURITY_DELAY</th><th>AIRLINE_DELAY</th><th>LATE_AIRCRAFT_DELAY</th><th>WEATHER_DELAY</th></tr><tr><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>cat</td><td>time</td><td>time</td><td>f64</td><td>f64</td><td>time</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td><td>time</td><td>f64</td><td>time</td><td>time</td><td>f64</td><td>i64</td><td>i64</td><td>cat</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;2015&quot;</td><td>&quot;2&quot;</td><td>&quot;1&quot;</td><td>&quot;7&quot;</td><td>&quot;NK&quot;</td><td>&quot;416&quot;</td><td>&quot;N627NK&quot;</td><td>&quot;FLL&quot;</td><td>&quot;IAG&quot;</td><td>22:15:00</td><td>22:06:00</td><td>-9.0</td><td>21.0</td><td>22:27:00</td><td>null</td><td>null</td><td>null</td><td>1176</td><td>null</td><td>null</td><td>01:07:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;B&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;2&quot;</td><td>&quot;10&quot;</td><td>&quot;2&quot;</td><td>&quot;NK&quot;</td><td>&quot;416&quot;</td><td>&quot;N632NK&quot;</td><td>&quot;FLL&quot;</td><td>&quot;IAG&quot;</td><td>22:15:00</td><td>22:06:00</td><td>-9.0</td><td>16.0</td><td>22:22:00</td><td>null</td><td>null</td><td>null</td><td>1176</td><td>null</td><td>null</td><td>01:07:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;B&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;4&quot;</td><td>&quot;20&quot;</td><td>&quot;1&quot;</td><td>&quot;NK&quot;</td><td>&quot;174&quot;</td><td>&quot;N630NK&quot;</td><td>&quot;FLL&quot;</td><td>&quot;LGA&quot;</td><td>16:02:00</td><td>18:57:00</td><td>175.0</td><td>24.0</td><td>19:21:00</td><td>null</td><td>null</td><td>null</td><td>1076</td><td>null</td><td>null</td><td>19:00:00</td><td>null</td><td>null</td><td>1</td><td>0</td><td>&quot;None&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;4&quot;</td><td>&quot;26&quot;</td><td>&quot;7&quot;</td><td>&quot;NK&quot;</td><td>&quot;720&quot;</td><td>&quot;N614NK&quot;</td><td>&quot;DEN&quot;</td><td>&quot;DFW&quot;</td><td>20:59:00</td><td>20:57:00</td><td>-2.0</td><td>13.0</td><td>21:10:00</td><td>null</td><td>null</td><td>null</td><td>641</td><td>null</td><td>null</td><td>23:50:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;B&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;5&quot;</td><td>&quot;9&quot;</td><td>&quot;6&quot;</td><td>&quot;NK&quot;</td><td>&quot;324&quot;</td><td>&quot;N620NK&quot;</td><td>&quot;MCO&quot;</td><td>&quot;ACY&quot;</td><td>21:30:00</td><td>21:52:00</td><td>22.0</td><td>18.0</td><td>22:10:00</td><td>null</td><td>null</td><td>null</td><td>852</td><td>null</td><td>null</td><td>23:40:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;B&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>&quot;2015&quot;</td><td>&quot;5&quot;</td><td>&quot;10&quot;</td><td>&quot;7&quot;</td><td>&quot;NK&quot;</td><td>&quot;368&quot;</td><td>&quot;N534NK&quot;</td><td>&quot;ORD&quot;</td><td>&quot;BWI&quot;</td><td>08:25:00</td><td>12:14:00</td><td>229.0</td><td>18.0</td><td>12:32:00</td><td>null</td><td>null</td><td>null</td><td>622</td><td>null</td><td>null</td><td>11:18:00</td><td>null</td><td>null</td><td>0</td><td>1</td><td>&quot;C&quot;</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></tbody></table></div>



As we can see, everything that should be inserted is there, besides scheduled time.

It can be imputed using mean for specific airports pair. Values will be calculated using only data from certain origin and destination and these means will be inserted in place of nulls.


```python
airports_w_nulls = (flights.filter(pl.col('SCHEDULED_TIME').is_null())
                    .select('ORIGIN_AIRPORT', 'DESTINATION_AIRPORT')
                    .unique()
                    .collect()
                    )
medians = (
    flights
    .join(
        airports_w_nulls.lazy(),
        on=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'],
        how='inner'
    )
    .filter(
        (~pl.col('SCHEDULED_TIME').is_null())
    )
    .group_by('ORIGIN_AIRPORT', 'DESTINATION_AIRPORT')
    .agg(pl.col("SCHEDULED_TIME")
         .mean()
         .round(1)
         .alias('times median'))
    .sort('times median', descending=False)
    .collect()
)

flights_imputed = (flights
    .join(
        medians.lazy(),
        on=['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'],
        how='left'
    )
    .with_columns(
    pl.col('SCHEDULED_TIME').fill_null(pl.col('times median'))
    )
    .drop('times median')
)
```

### Normal flights analysis

We should also look at the normal flights, if there are any nulls.


```python
print("No. nulls in airlines:\n",
      flights
      .filter((pl.col('CANCELLED') == 0),pl.col('DIVERTED')==0)
      .null_count()
      .unpivot(variable_name='tmp', value_name='null_count')
      .filter(pl.col('null_count') > 0)
      .collect())
```

    No. nulls in airlines:
     shape: (0, 2)
    ┌─────┬────────────┐
    │ tmp ┆ null_count │
    │ --- ┆ ---        │
    │ str ┆ u32        │
    ╞═════╪════════════╡
    └─────┴────────────┘
    

There are no nulls in these flights.

# Saving preprocessed dataset to parquet files


```python
flights.sink_parquet('data/flights.parquet')
airlines.sink_parquet('data/airlines.parquet')
airports.sink_parquet('data/airports.parquet')
```
