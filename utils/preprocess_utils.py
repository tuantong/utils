import pandas as pd

from forecasting.configs.logging_config import logger


def preprocess_stockout_data(
    df,
    id_col="id",
    target_col="y",
    is_stockout_col="is_stockout",
    window_size=4,
    min_periods=1,
):
    """
    Preprocesses stockout data for multiple items by filling demand during stockout periods
    based on the most recent non-stockout periods, ensuring that stockout values are
    replaced by the rolling mean of the last `window_size` valid non-stockout periods.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing stockout and demand data for multiple items.
    id_col : str, default 'id'
        Column name for the item identifier.
    target_col : str, default 'y'
        Column name for the demand/target variable.
    is_stockout_col : str, default 'is_stockout'
        Column name indicating stockout status (1 for stockout, 0 for non-stockout).
    window_size : int, default 4
        Size of the rolling window for calculating mean demand during non-stockout periods.
    min_periods : int, default 1
        Minimum number of observations required to calculate the rolling mean.

    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with filled stockout periods for all items.
    """
    logger.info(f"Starting preprocessing for {len(df[id_col].unique())} unique items")

    def process_single_item(group):
        stockout_mask = group[is_stockout_col] == 1
        stockout_count = stockout_mask.sum()

        # item_id = group[id_col].iloc[0]
        # logger.debug(f"Processing item {item_id}: {stockout_count} stockout periods")

        # If no stockout periods, return the group as-is
        if stockout_count == 0:
            # logger.debug(f"Item {item_id}: No stockout periods to process")
            return group

        # Handle non-stockout data
        non_stockout_data = group.loc[~stockout_mask, target_col]
        valid_non_stockout_data = non_stockout_data[non_stockout_data >= 0]

        if len(valid_non_stockout_data) == 0:
            # If no valid non-stockout periods, fallback to using the overall mean
            fill_value = group[target_col].mean()
            group.loc[stockout_mask, target_col] = fill_value
            return group

        # Calculate the rolling mean for non-stockout periods
        non_stockout_rolling_mean = valid_non_stockout_data.rolling(
            window=window_size, min_periods=min_periods
        ).mean()

        # Create a Series to store the fill values for stockout periods
        fill_values = pd.Series(index=group.index)

        # Iterate over stockout periods and fill based on prior `window_size` valid periods
        for start, end in get_stockout_periods(stockout_mask):
            # Try to use the last `window_size` valid non-stockout periods
            if start > 0:
                prior_data = group.loc[: start - 1, target_col]
                non_stockout_prior_data = prior_data[prior_data >= 0].tail(window_size)

                if not non_stockout_prior_data.empty:
                    # Use the mean of the last `window_size` valid periods
                    fill_value = non_stockout_prior_data.mean()
                else:
                    # If no prior data is found, use the first available rolling mean
                    fill_value = non_stockout_rolling_mean.iloc[0]
            else:
                # Stockout at the beginning, use the first available mean
                fill_value = non_stockout_rolling_mean.iloc[0]

            # Fill the stockout period with the calculated value
            fill_values.loc[start:end] = fill_value

        # Assign the fill values to the stockout periods in the group
        group.loc[stockout_mask, target_col] = fill_values[stockout_mask]

        return group

    # Create a copy of the dataframe to avoid modifying the original data
    df_processed = df.copy()

    # Process each item group separately and apply the logic
    return df_processed.groupby(id_col, group_keys=False).apply(process_single_item)


def get_stockout_periods(stockout_mask):
    """
    Identifies continuous stockout periods in the data.

    Parameters:
    -----------
    stockout_mask : pandas.Series
        Boolean series indicating stockout status (1 for stockout, 0 for non-stockout).

    Returns:
    --------
    list of tuples
        List of (start_index, end_index) for each stockout period.
    """
    stockout_indices = stockout_mask[stockout_mask].index
    if len(stockout_indices) == 0:
        return []

    periods = []
    start = stockout_indices[0]
    prev = start

    # Iterate through the stockout indices and find continuous periods
    for idx in stockout_indices[1:]:
        if idx != prev + 1:
            periods.append((start, prev))
            start = idx
        prev = idx

    # Append the last stockout period
    periods.append((start, prev))
    return periods
