import copy

import pandas as pd

from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler, merge_updated_data


def load_latest_data(config_dict, frequency="W-MON"):
    config_dict = copy.deepcopy(config_dict)
    config_dict["data"]["configs"]["freq"] = frequency
    full_data_handler = DataHandler(config_dict, subset="full")
    infer_data_handler = DataHandler(config_dict, subset="inference")
    full_df = full_data_handler.load_data()
    infer_df = infer_data_handler.load_data()
    full_df = merge_updated_data(full_df, infer_df)
    return full_df


def load_stock_data(config_dict):
    config_dict = copy.deepcopy(config_dict)
    config_dict["data"]["configs"]["freq"] = "D"
    full_data_handler = DataHandler(config_dict, subset="full")
    infer_data_handler = DataHandler(config_dict, subset="inference")
    stock_df = infer_data_handler.load_stock_data()
    stock_df = stock_df.astype(
        {
            "id": "string",
            "brand_name": "string",
            "platform": "string",
            "product_id": "string",
            "channel_id": "string",
            "date": "datetime64[us]",
            "stock": "float32",
            "variant_id": "string",
            "is_product": "bool",
            "is_stockout": "uint8",
        }
    )

    # Read and filter the daily data
    full_daily_df = pd.read_parquet(
        full_data_handler.dataset._metadata.source_data_path,
        engine="pyarrow",
        use_nullable_dtypes=True,
        filters=[
            ("date", ">=", stock_df["date"].min()),
        ],
    ).reset_index()
    infer_daily_df = pd.read_parquet(
        infer_data_handler.dataset._metadata.source_data_path,
        engine="pyarrow",
        use_nullable_dtypes=True,
        filters=[
            ("date", ">", full_daily_df["date"].max()),
        ],
    ).reset_index()

    latest_daily_df = pd.concat(
        [full_daily_df, infer_daily_df], ignore_index=True
    ).sort_values(["id", "date"])

    # Merge daily sales with stock data
    stock_df = stock_df.merge(
        latest_daily_df[["id", "date", "quantity_order"]], on=["id", "date"], how="left"
    ).dropna(subset=["id", "date", "quantity_order"])

    # Redefine stockout periods where both stock == 0 and quantity_order == 0
    stock_df["is_stockout"] = (
        (stock_df["stock"] == 0) & (stock_df["quantity_order"] == 0)
    ).astype(int)

    return stock_df[["id", "date", "quantity_order", "stock", "is_stockout"]]


def load_seasonal_set(config_dict, work_dir):
    full_data_handler = DataHandler(config_dict, subset="full")
    seasonal_set = full_data_handler.create_or_load_seasonal_item_list(work_dir)
    return seasonal_set


def split_data(time_series_df, meta_df, split_date, id_col="ID", date_col="ds"):
    if time_series_df.empty:
        raise ValueError("The input time series DataFrame is empty.")

    # Initial split based on the split_date
    train_df = time_series_df[time_series_df[date_col] < split_date]
    test_df = time_series_df[time_series_df[date_col] >= split_date]

    # Log initial shapes and unique IDs
    train_ids = train_df[id_col].unique()
    test_ids = test_df[id_col].unique()

    logger.info(f"train_df shape: {train_df.shape}, unique IDs: {train_ids.shape}")
    logger.info(f"test_df shape: {test_df.shape}, unique IDs: {test_ids.shape}")

    if train_df.empty:
        raise ValueError(
            "All data points are after the split date. No training data available."
        )

    if test_df.empty:
        logger.info(
            "No valid test data found. All available data will be used for training."
        )
        return time_series_df, test_df, meta_df

    # Identify IDs that appear only in one split
    train_id_set, test_id_set = set(train_ids), set(test_ids)

    only_in_train = train_id_set - test_id_set  # IDs with data only before split_date
    only_in_test = test_id_set - train_id_set  # IDs with data only after split_date

    # Log warnings for IDs removed
    if only_in_train:
        logger.warning(
            f"Removing {len(only_in_train)} IDs with data only in training set."
        )
        train_df = train_df[train_df[id_col].isin(test_ids)]
        meta_df = meta_df[meta_df[id_col].isin(test_ids)]

    if only_in_test:
        logger.warning(f"Removing {len(only_in_test)} IDs with data only in test set.")
        test_df = test_df[test_df[id_col].isin(train_ids)]
        meta_df = meta_df[meta_df[id_col].isin(train_ids)]

    # Log final shapes and unique IDs
    logger.info(
        f"Final train_df shape: {train_df.shape}, unique IDs: {train_df[id_col].unique().shape}"
    )
    logger.info(
        f"Final test_df shape: {test_df.shape}, unique IDs: {test_df[id_col].unique().shape}"
    )
    logger.info(
        f"Final meta_df shape: {meta_df.shape}, unique IDs: {meta_df[id_col].unique().shape}"
    )

    logger.info(f"train max date: {train_df[date_col].max()}")
    logger.info(
        f"test min date: {test_df[date_col].min()}, test max date: {test_df[date_col].max()}"
    )
    return train_df, test_df, meta_df
