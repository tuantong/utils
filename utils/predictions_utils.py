import pandas as pd
from dateutil.relativedelta import MO, relativedelta

from forecasting.configs.logging_config import logger
from forecasting.data.util import (
    calc_created_time,
    calc_sale_per_day,
    remove_leading_zeros,
)
from forecasting.evaluation.aggregation_utils import (
    aggregate_daily_ts_to_monthly,
    aggregate_monthly_ts_to_daily,
    aggregate_weekly_ts_to_daily,
    aggregate_weekly_ts_to_monthly,
    build_result_df_from_pd,
)


def prepare_prediction_df(
    train_df,
    pred_df,
    meta_df,
    freq,
    target_col="quantity_order",
    prediction_col="yhat",
    id_col="id",
    date_col="date",
):
    logger.info("Preparing predictions data...")
    # Rename `id_col` column to "id" if not already
    if "id" not in pred_df.columns:
        pred_df = pred_df.rename(columns={id_col: "id"})
    if "id" not in train_df.columns:
        train_df = train_df.rename(columns={id_col: "id"})

    # Rename `date_col` column to "date" if not already
    if "date" not in pred_df.columns:
        pred_df = pred_df.rename(columns={date_col: "date"})
    if "date" not in train_df.columns:
        train_df = train_df.rename(columns={date_col: "date"})

    pred_pivot_df = build_result_df_from_pd(
        train_df, pred_df, pred_cols=[prediction_col], target_col=target_col
    )
    # Merge metadata
    pred_pivot_df = pred_pivot_df.merge(meta_df, on="id", how="left")

    # Find inference date
    inference_date = pd.to_datetime(pred_df[date_col].min())

    # If freq is W-MON, get the previous monday from the `inference_date` if it not already monday
    # If freq is W, get first day of month from `inference_date`
    first_daily_pred_date = (
        inference_date.date() - relativedelta(weekday=MO(-1))
        if freq == "W-MON"
        else inference_date.replace(day=1).date()
    )
    logger.info(f"First daily inference date: {first_daily_pred_date}")

    # Create daily_train_ts column to calculate sale_per_day
    pred_pivot_df["daily_train_ts"] = pred_pivot_df.apply(
        lambda x: (
            aggregate_weekly_ts_to_daily(x["train_ts"])
            if freq == "W-MON"
            else aggregate_monthly_ts_to_daily(
                x["train_ts"], pd.to_datetime(x["first_train_date"]).replace(day=1)
            )
        ),
        axis=1,
    )

    # Create daily_pred_ts column
    pred_pivot_df["daily_pred_ts"] = pred_pivot_df[prediction_col].apply(
        lambda x: (
            aggregate_weekly_ts_to_daily(x)
            if freq == "W-MON"
            else aggregate_monthly_ts_to_daily(x, first_daily_pred_date)
        )
    )

    # Remove leading zeros of every time series to calculate sale_per_day
    pred_pivot_df = pred_pivot_df.assign(
        # Create daily_train_ts_cut
        daily_train_ts_cut=(
            pred_pivot_df.daily_train_ts.apply(lambda x: remove_leading_zeros(x))
        )
    )

    pred_pivot_df["sale_per_day"] = pred_pivot_df.daily_train_ts_cut.apply(
        lambda x: calc_sale_per_day(x)
    )

    pred_pivot_df = pred_pivot_df.assign(
        # Calculate created_time
        created_time=(
            pred_pivot_df.created_date.apply(
                lambda x: calc_created_time(first_daily_pred_date, x)
            )
        )
    ).assign(
        # Calculate length of monthly historical sales after removing leading zeros
        item_length=(
            pred_pivot_df.monthly_train_ts.apply(remove_leading_zeros).apply(len)
        )
    )

    # # Create 'monthly_pred_ts'
    # pred_pivot_df = pred_pivot_df.assign(
    #     **{
    #         "monthly_pred_ts": pred_pivot_df.apply(
    #             lambda x: aggregate_daily_ts_to_monthly(
    #                 x["daily_pred_ts"], x["first_pred_date"]
    #             ),
    #             axis=1,
    #         )
    #     }
    # )

    # # Truncate 'monthly_pred_ts' to keep only next 14 months
    # pred_pivot_df["monthly_pred_ts"] = pred_pivot_df.monthly_pred_ts.apply(
    #     lambda x: x[:14]
    # )

    # Recreate 'daily_pred_ts' from 'monthly_pred_ts'
    # pred_pivot_df = pred_pivot_df.assign(
    #     **{
    #         "daily_pred_ts": pred_pivot_df.apply(
    #             lambda x: (
    #                 aggregate_monthly_ts_to_daily(
    #                     x["monthly_pred_ts"],
    #                     first_daily_pred_date,
    #                 )
    #             ),
    #             axis=1,
    #         )
    #     }
    # )

    # Truncate `daily_pred_ts` to have the same daily prediction length for all items
    min_len_daily_pred_ts = min(
        [
            len(pred_pivot_df.daily_pred_ts.values[i])
            for i in range(pred_pivot_df.shape[0])
        ]
    )

    pred_pivot_df["daily_pred_ts"] = pred_pivot_df.daily_pred_ts.apply(
        lambda x: x[:min_len_daily_pred_ts]
    )

    logger.info(
        f"pred_pivot_df shape: {pred_pivot_df.shape}, {pred_pivot_df['id'].unique().shape}"
    )

    # Create 'monthly_pred_ts'
    pred_pivot_df = pred_pivot_df.assign(
        **{
            "monthly_pred_ts": pred_pivot_df.apply(
                lambda x: aggregate_daily_ts_to_monthly(
                    x["daily_pred_ts"], first_daily_pred_date
                ),
                axis=1,
            )
        }
    )

    return pred_pivot_df


def extract_yhat(forecasts: pd.DataFrame):
    """Extract 'yhat' from a forecast DataFrame by coalescing all 'yhat_i' columns"""
    # Extract all 'yhat' columns and prioritize values from left to right
    yhats_columns = [col for col in forecasts.columns if "yhat" in col]

    # Copy forecasts to avoid modifying the original DataFrame
    forecasts_copy = forecasts.copy()

    # Create a new 'yhat' column by coalescing all 'yhat_i' columns, prioritizing non-null values
    forecasts_copy["yhat"] = forecasts_copy[yhats_columns].bfill(axis=1).iloc[:, 0]

    # Filter out rows where 'yhat' is still missing and select relevant columns
    return (
        forecasts_copy[["ID", "ds", "yhat"]]
        .dropna(subset=["yhat"])
        .reset_index(drop=True)
    )


def combine_forecasts(
    forecast1,
    forecast2,
    n_step,
    fh_m1,
    fh_m2,
    freq_m1,
    freq_m2,
    first_pred_date,
    common_freq=None,
):
    """
    Combine forecasts from model1 and model2 based on the cutoff point n_step.

    Parameters:
    - forecast1: Forecast DataFrame from model1
    - forecast2: Forecast DataFrame from model2
    - n_step: The cutoff point at which to switch from model1 to model2
    - fh_m1: Forecast horizon for model1
    - fh_m2: Forecast horizon for model2
    - freq_m1: Original frequency of forecast1 ("W-MON" or "M")
    - freq_m2: Original frequency of forecast2 ("W-MON" or "M")
    - first_pred_date: The date of the first prediction in the combined forecast
    - common_freq: The frequency to which both forecasts should be resampled (default: None)

    Returns:
    - Combined DataFrame with forecasts from model1 and model2.
    """
    # Handle different forecast horizon scenarios
    if freq_m1 == freq_m2 and fh_m1 > fh_m2:
        raise ValueError(
            f"Error: Model 1's forecast horizon {fh_m1} is greater than Model 2's {fh_m2}."
        )

    if common_freq is not None and freq_m1 == "M":
        # Convert n_step to weeks
        n_step = n_step * 4

    def resample_to_common_frequency(
        forecast, original_freq, target_freq, first_pred_date
    ):
        """Resample forecast from 'W-MON' or 'M' to the target frequency."""
        if original_freq == "W-MON" and target_freq == "M":
            resampled_ts = aggregate_weekly_ts_to_monthly(
                forecast["yhat"].values, forecast["ds"].iloc[0]
            )
            return pd.DataFrame(
                {
                    "ID": forecast["ID"].iloc[0],
                    "ds": pd.date_range(
                        forecast["ds"].iloc[0], periods=len(resampled_ts), freq="M"
                    ),
                    "yhat": resampled_ts,
                }
            )

        elif original_freq == "M" and target_freq == "W-MON":
            first_day_of_month = first_pred_date.strftime("%Y-%m-01")
            daily_ts = aggregate_monthly_ts_to_daily(
                forecast["yhat"].values, first_day_of_month
            )
            daily_df = pd.DataFrame(
                {
                    "ID": forecast["ID"].iloc[0],
                    "ds": pd.date_range(
                        first_day_of_month, periods=len(daily_ts), freq="D"
                    ),
                    "yhat": daily_ts,
                }
            )
            resampled_df = daily_df.resample(
                "W-MON", on="ds", closed="left", label="left"
            ).agg({"ID": "first", "yhat": "sum"})
            return resampled_df.loc[resampled_df.index >= first_pred_date].reset_index()

        # If no resampling is needed, return the original forecast
        return forecast

    # Extract yhat columns from both forecasts
    forecast1_extracted = extract_yhat(forecast1)
    forecast2_extracted = extract_yhat(forecast2)

    # Extract future forecast only from both forecasts
    forecast1_extracted = forecast1_extracted[
        forecast1_extracted["ds"] >= first_pred_date
    ]
    forecast2_extracted = forecast2_extracted[
        forecast2_extracted["ds"] >= first_pred_date
    ]

    # Resample forecasts to the common frequency
    # TODO: Handle later
    if resample_to_common_frequency is not None:
        forecast1_extracted = resample_to_common_frequency(
            forecast1_extracted, freq_m1, first_pred_date, common_freq
        )
        forecast2_extracted = resample_to_common_frequency(
            forecast2_extracted, freq_m2, first_pred_date, common_freq
        )

    # Slice forecasts: Use first n_step from model1, and the rest from model2
    forecast1_extracted = forecast1_extracted.iloc[:n_step]
    forecast2_extracted = forecast2_extracted.iloc[n_step:]

    combined_forecast = pd.concat(
        [forecast1_extracted, forecast2_extracted], axis=0, ignore_index=True
    )

    return combined_forecast
