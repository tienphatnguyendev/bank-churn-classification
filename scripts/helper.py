import pandas as pd
import re


def convert_to_snake_case(column):
    if column == "CustomerID":
        column = "customer_id"
    snake_case = re.sub(r"(?<!^)([A-Z])", r"_\1", column)
    snake_case = snake_case.lower()
    return snake_case


def rename_col(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.columns = [convert_to_snake_case(col) for col in df.columns]
        return df
    except Exception as e:
        print(e)
