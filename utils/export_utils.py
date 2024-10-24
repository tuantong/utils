import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.monitor.forecast_monitor import ForecastMonitor
from forecasting.util import NpEncoder

BRANDS_TO_CREATE_ALL_CHANNEL = ["mizmooz", "as98"]


def export_result(
    final_agg_result_df,
    result_save_dir,
    brand_name,
    forecast_date,
    level_list=["variant", "product"],
):
    # Generate and save results to json
    logger.info(f"Generating results for {brand_name}...")
    brand_df = final_agg_result_df[final_agg_result_df.brand_name == brand_name]

    # Map variant_id with product_id
    variant_df = brand_df[brand_df.is_product == False].drop_duplicates(
        subset=["variant_id"]
    )
    dict_variant_product = dict(zip(variant_df.variant_id, variant_df.product_id))

    # Calculate accuracy_score
    logger.info(f"Monitor accuracy for {brand_name}")
    forecast_monitor = ForecastMonitor(brand_name)
    # Accuracy last_3_months for confidence_score
    results = forecast_monitor.get_forecast_accuracy_all_items(
        period="3 months",
        method="mape",
        group_method="sale_category",
        forecast_date=forecast_date,
    )

    if results is not None:
        error_results = results["error_results"]
        logger.info(f"Len of last_3_months_acc_result: {len(error_results)}")
    else:
        logger.info("No have enough forecast for monitoring by month")
        error_results = None

    brand_save_path = Path(result_save_dir) / brand_name
    os.makedirs(brand_save_path, exist_ok=True)

    for level in level_list:
        result_list = []
        if level == "variant":
            field_name = "variant_id"
            full_level_df = brand_df[brand_df.is_product == False]
            full_level_df["item_id"] = full_level_df.variant_id
        else:
            field_name = "product_id"
            full_level_df = brand_df[brand_df.is_product == True]
            full_level_df["item_id"] = full_level_df.product_id
        logger.info(f"Number of unique {level} IDs:{full_level_df.id.unique().shape}")
        logger.info(f"Shape of {level} level's dataframe: {full_level_df.shape}")

        for row in tqdm(
            full_level_df.itertuples(),
            total=full_level_df.shape[0],
            desc=f"Generating results of {level} level...",
        ):
            result = {
                field_name: row.item_id,
                "h_key": row.h_key if pd.isna(row.h_key) == False else None,
                "from_source": row.platform,
                "channel_id": row.channel_id,
                "forecast_date": str(forecast_date),
                "weekly_historical_val": str(row.train_ts),
                "monthly_historical_val": str(row.monthly_train_ts),
                "monthly_prediction_val": str(np.ceil(row.monthly_pred_ts).tolist()),
                "predictions": {
                    "sale_per_day": row.sale_per_day,
                    "forecast_val": str(row.daily_pred_ts),
                    "trend": None,
                },
            }
            if (error_results is not None) and (row.id in error_results.keys()):
                result["sale_pattern"] = error_results[row.id]["sale_pattern"]
                result["confidence_score"] = error_results[row.id]["confidence_score"]
            else:
                result["sale_pattern"] = None
                result["confidence_score"] = None

            # Append result into the list of results
            result_list.append(result)
        logger.info(f"Number of {level} results: {len(result_list)}")

        # Sum forecast all_channel
        # if "0" not in full_level_df.channel_id.unique():
        if brand_name in BRANDS_TO_CREATE_ALL_CHANNEL:
            logger.info(f"Generate forecast result for {brand_name} all_channel...")
            ids_list = list({result[field_name] for result in result_list})
            for item in tqdm(ids_list, total=len(ids_list)):
                item_results = [res for res in result_list if res[field_name] == item]
                weekly_history_list = [
                    ast.literal_eval(res["weekly_historical_val"])
                    for res in item_results
                ]
                len_weekly_history = min([len(x) for x in weekly_history_list])
                weekly_history_list = [
                    x[-len_weekly_history:] for x in weekly_history_list
                ]
                monthly_history_list = [
                    ast.literal_eval(res["monthly_historical_val"])
                    for res in item_results
                ]
                len_history = min([len(x) for x in monthly_history_list])
                monthly_history_list = [x[-len_history:] for x in monthly_history_list]
                monthly_prediction_list = [
                    ast.literal_eval(res["monthly_prediction_val"])
                    for res in item_results
                ]
                sale_per_day_list = [
                    res["predictions"]["sale_per_day"] for res in item_results
                ]
                forecast_list = [
                    ast.literal_eval(res["predictions"]["forecast_val"])
                    for res in item_results
                ]
                platform = item_results[0]["from_source"]
                h_key = item_results[0]["h_key"]

                if level == "product":
                    unique_id = brand_name + "_" + platform + "_" + item + "_NA_0"
                else:
                    unique_id = (
                        brand_name
                        + "_"
                        + platform
                        + "_"
                        + dict_variant_product[item]
                        + "_"
                        + item
                        + "_0"
                    )

                all_channel_result = {
                    field_name: item,
                    "h_key": h_key,
                    "from_source": platform,
                    "channel_id": "0",
                    "forecast_date": str(forecast_date),
                    "weekly_historical_val": str(
                        [round(sum(x), 2) for x in zip(*weekly_history_list)]
                    ),
                    "monthly_historical_val": str(
                        [round(sum(x), 2) for x in zip(*monthly_history_list)]
                    ),
                    "monthly_prediction_val": str(
                        [round(sum(x), 2) for x in zip(*monthly_prediction_list)]
                    ),
                    "predictions": {
                        "sale_per_day": round(np.sum(sale_per_day_list), 7),
                        "forecast_val": str(
                            [round(sum(x), 7) for x in zip(*forecast_list)]
                        ),
                        "trend": None,
                    },
                }
                if (error_results is not None) and (unique_id in error_results.keys()):
                    all_channel_result["sale_pattern"] = error_results[unique_id][
                        "sale_pattern"
                    ]
                    all_channel_result["confidence_score"] = error_results[unique_id][
                        "confidence_score"
                    ]
                else:
                    all_channel_result["sale_pattern"] = None
                    all_channel_result["confidence_score"] = None

                result_list.append(all_channel_result)
            logger.info(
                f"Number of {level} results after generating all_channel results: {len(result_list)}"
            )

        # Saving results
        logger.info(f"Saving results for {level} level...")
        save_path = Path(brand_save_path) / f"{level}_result_forecast.json"
        try:
            with open(save_path, "w") as file:
                json.dump(result_list, file, cls=NpEncoder, indent=4)
        except Exception as err:
            logger.exception("An error occured while saving: ", err)
            raise
        else:
            logger.info(
                f"Successfully saved results to {Path(brand_save_path).absolute()}"
            )
