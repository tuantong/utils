import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.evaluation.aggregation_utils import (
    aggregate_bottom_up,
    aggregate_daily_ts_to_weekly,
    aggregate_monthly_ts_to_daily,
    aggregate_monthly_ts_to_weekly,
    aggregate_weekly_ts_to_daily,
    clip_channel_pred_smaller_than_all_channel_pred,
)
from forecasting.util import get_formatted_duration
from inference.utils import (
    adjust_forecast_to_last_year_pattern,
    fill_stockout,
    last_date_of_month,
)


def calculate_scale_ratio(item_monthly_hist, similar_items_hist, item_created_length):
    item_created_time_hist_sum = item_monthly_hist[-item_created_length:].sum()
    similar_items_hist_mean = (
        similar_items_hist[:, -item_created_length:].sum(axis=1).mean()
    )
    scale_ratio = (
        item_created_time_hist_sum / similar_items_hist_mean
        if similar_items_hist_mean != 0
        else 1
    )
    return scale_ratio


def compute_forecast(similar_items_preds, scale_ratio, item_monthly_hist):
    monthly_raw_avg_sim_forecast = np.mean(similar_items_preds, axis=0)
    new_ratio = 1.5 * item_monthly_hist.max() / monthly_raw_avg_sim_forecast.max()
    final_ratio = min(scale_ratio, new_ratio)

    monthly_raw_avg_sim_forecast_scaled = np.round(
        monthly_raw_avg_sim_forecast * final_ratio
    )

    pred_ts = monthly_raw_avg_sim_forecast_scaled[:14]
    return pred_ts


def prepare_time_range_and_plot_data(item_first_pred_date, item_monthly_hist, pred_ts):
    thresh_date = last_date_of_month(item_first_pred_date)
    if item_first_pred_date.date().day != 1:
        start_date = thresh_date + relativedelta(months=-len(item_monthly_hist) + 1)
    else:
        start_date = thresh_date + relativedelta(months=-len(item_monthly_hist))

    end_date = thresh_date + relativedelta(months=+len(pred_ts) - 1)

    plot_period = pd.period_range(start_date, end_date, freq="M")

    hist_df = pd.DataFrame(
        {"date": plot_period[: len(item_monthly_hist)], "values": item_monthly_hist}
    )
    pred_df = pd.DataFrame({"date": plot_period[-len(pred_ts) :], "values": pred_ts})

    return hist_df, pred_df


def adjust_product_forecast_from_similar_items_avg_w_scale_ratio_and_historical_pattern(
    product_level_df,
    similar_items_pred_df,
    monthly_pred_col="monthly_pred_ts",
    monthly_train_col="monthly_train_ts",
    first_pred_date_col="first_pred_date",
    item_length_col="item_length",
    created_time_col="created_time",
    similar_items_col="similar_items",
    truncate_min_daily_len=None,
):
    adjust_pred_dict = {}

    for item_pred in tqdm(
        product_level_df.itertuples(),
        total=product_level_df.shape[0],
        desc="Adjusting products forecast based on similar products",
    ):
        item_id = item_pred.id
        similar_items_of_item = getattr(item_pred, similar_items_col)
        item_first_pred_date = getattr(item_pred, first_pred_date_col)
        item_created_length = getattr(item_pred, item_length_col)
        item_created_time = getattr(item_pred, created_time_col)

        if similar_items_of_item is not None:
            similar_items_preds = np.stack(
                similar_items_pred_df[monthly_pred_col].apply(np.array), axis=0
            )

            item_monthly_hist = np.array(getattr(item_pred, monthly_train_col))
            if (item_created_time > 14) or (item_monthly_hist.max() > 0):
                min_len_similar_items = (
                    similar_items_pred_df[monthly_train_col].apply(len).min()
                )
                similar_items_hist = np.stack(
                    similar_items_pred_df[monthly_train_col]
                    .apply(np.array)
                    .apply(lambda x: x[-min_len_similar_items:]),
                    axis=0,
                )

                scale_ratio = calculate_scale_ratio(
                    item_monthly_hist, similar_items_hist, item_created_length
                )

                pred_ts = compute_forecast(
                    similar_items_preds, scale_ratio, item_monthly_hist
                )

                hist_df, pred_df = prepare_time_range_and_plot_data(
                    item_first_pred_date, item_monthly_hist, pred_ts
                )

                monthly_pred_adjusted = adjust_forecast_to_last_year_pattern(
                    hist_df, pred_df
                )

                daily_pred_adjusted = aggregate_monthly_ts_to_daily(
                    monthly_pred_adjusted["values"].tolist(), item_first_pred_date
                )
            else:
                monthly_pred_adjusted = np.mean(similar_items_preds, axis=0)
                daily_pred_adjusted = aggregate_monthly_ts_to_daily(
                    monthly_pred_adjusted.tolist(), item_first_pred_date
                )
            if truncate_min_daily_len is not None:
                daily_pred_adjusted = daily_pred_adjusted[:truncate_min_daily_len]
            adjust_pred_dict[item_id] = daily_pred_adjusted

    return adjust_pred_dict


def post_process_stockout_data(
    agg_result_df, brand_config_dict, forecast_date, forecast_range
):
    start_stockout = time.time()

    def prepare_stockout_data():
        logger.info("Process stockout dataset...")

        date_forecast_range = [date.strftime("%m-%d") for date in forecast_range]
        date_last_year = pd.to_datetime(forecast_date - relativedelta(years=1))
        stockout_df = pd.DataFrame()
        brand_infer_data_handler = DataHandler(brand_config_dict, subset="inference")
        brand_stock_df = brand_infer_data_handler.load_stock_data()
        brand_stock_df = brand_stock_df[
            (brand_stock_df.is_product == False)
            & (
                brand_stock_df.date.between(
                    date_last_year, pd.to_datetime(forecast_date)
                )
            )
        ]

        brand_stockout_df = brand_stock_df[brand_stock_df.is_stockout == 1]
        brand_stockout_df["stockout_id"] = brand_stockout_df.apply(
            lambda row: f"{row['platform']}_{row['variant_id']}", axis=1
        )
        brand_stockout_df["date"] = brand_stockout_df.date.apply(
            lambda x: x.strftime("%m-%d")
        )
        item_list = brand_stockout_df.stockout_id.unique().tolist()
        brand_stockout_df = brand_stockout_df.set_index(["date", "stockout_id"])
        brand_stockout_df = brand_stockout_df[~brand_stockout_df.index.duplicated()]

        multi_index = pd.MultiIndex.from_product([date_forecast_range, item_list])
        multi_index = multi_index.set_names(["date", "stockout_id"])
        brand_stockout_df = brand_stockout_df.reindex(multi_index).reset_index()
        brand_stockout_df["brand_name"] = brand_config_dict["data"]["configs"]["name"]

        stockout_df = pd.concat([stockout_df, brand_stockout_df])

        return stockout_df

    stockout_df = prepare_stockout_data()
    if stockout_df.shape[0] > 0:
        stockout_df = stockout_df.reset_index(drop=True)

        agg_product_df = agg_result_df[agg_result_df.is_product == True]
        agg_variant_df = agg_result_df[agg_result_df.is_product == False]

        # Process and merge stockout_ts to agg_variant_df
        stockout_df.is_stockout = stockout_df.is_stockout.fillna(value=0)
        stockout_df.is_stockout = stockout_df.is_stockout.astype(float)

        pivot_stockout_df = pd.pivot_table(
            stockout_df,
            index=["brand_name", "stockout_id"],
            values=["is_stockout"],
            aggfunc=lambda x: list(x),
        ).reset_index()
        agg_variant_df["stockout_id"] = agg_variant_df.apply(
            lambda row: f"{row['platform']}_{row['variant_id']}", axis=1
        )
        agg_variant_df = agg_variant_df.merge(
            pivot_stockout_df, on=["brand_name", "stockout_id"], how="left"
        )
        stockout_item_list = pivot_stockout_df.set_index(
            ["brand_name", "stockout_id"]
        ).index.tolist()
        logger.info(f"Number of variant with stockout: {len(stockout_item_list)}")

        # Run filling stockout daily_pred_ts for variant level
        logger.info(
            "Fill stockout daily_pred_ts with average quantity over previous days..."
        )
        agg_variant_df["daily_pred_ts"] = agg_variant_df.apply(
            lambda row: (
                fill_stockout(
                    stockout_ts=row.is_stockout,
                    daily_train_ts=row.daily_train_ts,
                    daily_pred_ts=row.daily_pred_ts,
                    item_id=row.id,
                )
                if (row.brand_name, row.stockout_id) in stockout_item_list
                else row.daily_pred_ts
            ),
            axis=1,
        )
        logger.info(
            f"Time for filling stockout predictions: {get_formatted_duration(time.time() - start_stockout)}"
        )
        stockout_result_df = pd.concat([agg_product_df, agg_variant_df])

        # Check again if online_pred > all_channel_pred
        final_stockout_result_df = clip_channel_pred_smaller_than_all_channel_pred(
            stockout_result_df
        )
        # final_stockout_result_df["daily_pred_ts"] = final_stockout_result_df.apply(
        #     lambda row: tweak_variant(row["daily_pred_ts"], row["id"]), axis=1
        # )

        logger.info("Aggregate bottom-up again...")
        final_agg_result_df = aggregate_bottom_up(
            final_stockout_result_df, pred_column="daily_pred_ts"
        )
    else:
        final_agg_result_df = agg_result_df
    return final_agg_result_df


def extract_bounds(hist_array: np.array):
    # Extract last 3 months and 12 months data
    last_3_months_data = hist_array[-12:]
    last_12_months_data = hist_array[-52:]
    last_3_months_data_sum_month = [
        sum(last_3_months_data[i : i + 4]) for i in range(0, len(last_3_months_data), 4)
    ]
    last_12_months_data_sum_month = [
        sum(last_12_months_data[i : i + 4])
        for i in range(0, len(last_12_months_data), 4)
    ]

    # Calculate constraints
    max_history_3m = max(last_3_months_data_sum_month) / 4
    avg_sale_3m = last_3_months_data.mean()

    max_history_12m = max(last_12_months_data_sum_month) / 4
    avg_sale_12m = last_12_months_data.mean()

    # Define upper and lower bounds for both models
    if len(hist_array) < 52:
        upper_bound_3m = 1.5 * max_history_3m
    else:
        upper_bound_3m = 3.0 * max_history_3m

    lower_bound_3m = 0.5 * avg_sale_3m

    upper_bound_12m = 1.5 * max_history_12m
    lower_bound_12m = 0.5 * avg_sale_12m
    return lower_bound_3m, upper_bound_3m, lower_bound_12m, upper_bound_12m


def soft_clip(x, lower_bound, upper_bound, smoothness=0.5):
    def sigmoid(t):
        return 1 / (1 + np.exp(-t / smoothness))

    lower_clip = lower_bound + (x - lower_bound) * sigmoid(x - lower_bound)
    upper_clip = upper_bound - (upper_bound - lower_clip) * sigmoid(
        upper_bound - lower_clip
    )
    return upper_clip


def post_process_soft_clip(df_pivot, freq, daily_pred_col="daily_pred_ts"):
    adjust_pred_dict = {}
    for item_pred in tqdm(
        df_pivot.itertuples(), total=df_pivot.shape[0], desc="Soft clipping predictions"
    ):
        item_id = item_pred.id
        logger.debug(f"Performing soft clipping for {item_id}...")
        # Convert daily predictions to weekly
        first_pred_date = (
            item_pred.first_pred_date
            if freq == "W-MON"
            else item_pred.first_pred_date.replace(day=1)
        )
        item_weekly_pred_ts = np.array(
            aggregate_daily_ts_to_weekly(
                np.array(getattr(item_pred, daily_pred_col)), first_pred_date
            )
        )
        logger.debug(f"item train_ts: {getattr(item_pred, 'train_ts')}")
        logger.debug(f"yhat: {getattr(item_pred, 'yhat')}")
        logger.debug(f"item_weekly_pred_ts before soft clipping: {item_weekly_pred_ts}")

        # Extract bounds from weekly history
        # train_ts could be weekly or monthly so we need to convert it to weekly
        train_ts = np.array(getattr(item_pred, "train_ts"))
        first_train_date = pd.to_datetime(item_pred.first_train_date).replace(day=1)
        train_ts = (
            train_ts
            if freq == "W-MON"
            else np.array(aggregate_monthly_ts_to_weekly(train_ts, first_train_date))
        )
        lower_bound_3m, upper_bound_3m, lower_bound_12m, upper_bound_12m = (
            extract_bounds(train_ts)
        )
        logger.debug(
            f"lower_bound_3m: {lower_bound_3m}, upper_bound_3m: {upper_bound_3m}, "
            f"lower_bound_12m: {lower_bound_12m}, upper_bound_12m: {upper_bound_12m}"
        )
        # Perform soft clipping
        item_weekly_pred_ts[:12] = soft_clip(
            item_weekly_pred_ts[:12], lower_bound_3m, upper_bound_3m, 0.5
        )
        item_weekly_pred_ts[12:] = soft_clip(
            item_weekly_pred_ts[12:], lower_bound_12m, upper_bound_12m, 0.5
        )

        logger.debug(f"item_weekly_pred_ts after soft clipping: {item_weekly_pred_ts}")
        # Convert the weekly predictions back to daily
        item_daily_pred_ts = aggregate_weekly_ts_to_daily(item_weekly_pred_ts)
        adjust_pred_dict[item_id] = item_daily_pred_ts

    return adjust_pred_dict
