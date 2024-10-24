import time

from forecasting.configs.logging_config import logger
from forecasting.evaluation.aggregation_utils import (
    aggregate_top_down_based_on_sale_distribution,
    clip_channel_pred_smaller_than_all_channel_pred,
)
from forecasting.util import get_formatted_duration


def run_top_down_disaggregation(pred_pivot_df, channel_list):
    # Run top-down disaggregation
    start_filter = time.time()
    pred_pivot_df = pred_pivot_df[pred_pivot_df.channel_id.isin(channel_list)]
    logger.info(
        f"Full_pred_df after filtering channel_id: {pred_pivot_df.shape[0]}, {pred_pivot_df.id.unique().shape}"
    )

    logger.info("Clip pred_ts online channel <= all_channel...")
    final_full_pred_df = clip_channel_pred_smaller_than_all_channel_pred(pred_pivot_df)
    logger.info(
        f"Final_full_pred_df after clip online_channel pred_ts: {final_full_pred_df.shape[0]}, {final_full_pred_df.id.unique().shape}"
    )

    agg_result_df = aggregate_top_down_based_on_sale_distribution(
        final_full_pred_df, pred_column="daily_pred_ts"
    )
    logger.info(f"Full_pred_df after aggregation top-down: {agg_result_df.shape[0]}")
    logger.info(
        f"Time for filtering channel and aggregate top-down: {get_formatted_duration(time.time() - start_filter)}"
    )
    return agg_result_df
