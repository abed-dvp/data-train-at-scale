from taxifare.utils import simple_time_and_memory_tracker
import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *

@simple_time_and_memory_tracker
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    # Compress raw_data by setting types to DTYPES_RAW
    # YOUR CODE HERE
    df = df.astype(DTYPES_RAW)

    # Remove buggy transactions
    # YOUR CODE HERE
    df = df.drop_duplicates()
    df = df.dropna(how="any", axis=0)

    df = df[df["passenger_count"] > 0]
    df = df[df["fare_amount"] > 0]

    # Remove geographically irrelevant transactions (rows)
    # YOUR CODE HERE
    df = df[df["pickup_latitude"].between(40.5, 40.9)]
    df = df[df["dropoff_latitude"].between(40.5, 40.9)]

    df = df[df["pickup_longitude"].between(-74.3, -73.7)]
    df = df[df["dropoff_longitude"].between(-74.3, -73.7)]

    df = df[df["fare_amount"] < 400]
    df = df[df["passenger_count"] < 8]

    print("✅ data cleaned")

    return df
