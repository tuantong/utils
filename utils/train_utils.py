import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from omegaconf import DictConfig
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.data.util import find_leading_zeros
from forecasting.util import TqdmToLoguru
from forecasting.utils.common_utils import save_dict_to_json
from utils.config_utils import (
    determine_model_configs,
    parse_data_configs,
    parse_model_configs,
)
from utils.model_utils import set_seeds
from utils.predictions_utils import combine_forecasts

######## custom-event  ##########
def decompose_df_events(df_id, df_events):
    df_events['ds'] = pd.to_datetime(df_events['ds'])
    cut_off_date = df_id['ds'].iloc[-1]
    df_events_past = df_events[df_events['ds'] <= cut_off_date]
    df_events_future = df_events[df_events['ds'] > cut_off_date]
    common =  list(set(df_events_future['event']) & set(df_events_past['event']))
    df_events_future_train = df_events_future[df_events_future['event'].isin(common)]
    df_events_trains = [df_events_past, df_events_future_train]
    df_events_train = pd.concat(df_events_trains, ignore_index=True)
    df_events_untrain = df_events_future[~df_events_future['event'].isin(common)]
    return df_events_train, df_events_untrain

def evented_forecast(forecast, df_events_untrain):
    forecast_evented = forecast.merge(df_events_untrain[['ds','impactValue']], on='ds', how='left')
    forecast_evented['impactValue'] = forecast_evented['impactValue'].fillna(0)
    forecast_evented['yhat'] = forecast_evented['yhat'] * (1 + forecast_evented['impactValue'])
    return forecast_evented
######## custom-event  ##########

def fit_and_predict_chunks(
    cfg: DictConfig,
    df: pd.DataFrame,
    first_pred_date: Optional[Union[str, datetime.datetime]],
    work_dir: str,
    id_col: str = "id",
    target_col: str = "quantity_order",
    date_col: str = "date",
    num_workers: int = 4,
    log_interval: int = 50,
    df_events_all: pd.DataFrame = None, # custom-event
):
    if isinstance(first_pred_date, str):
        first_pred_date = pd.to_datetime(first_pred_date)
        # # Get the next monday from the max date in the group
        # first_pred_date = df_group["ds"].max() + relativedelta(weekday=MO(2))

    def process_group(group):
        mask = find_leading_zeros(group[target_col])
        group_cleaned = group[~mask]

        # BUG: After removing leading zeros, if the time series contains only one unique value, we will
        # encounter the error: `Encountered variable with singular value in training set. Please remove variable.`
        # when training Neural Prophet
        # Temporary fix: add 1 to the first value
        if len(np.unique(group_cleaned[target_col])) == 1:
            group_cleaned.loc[group_cleaned.index[0], target_col] += 1
        return group_cleaned

    id_list = df[id_col].unique().tolist()
    logger.info(f"Total number of IDs: {len(id_list)}")

    df_cleaned = (
        df.groupby(id_col, group_keys=False).apply(process_group).reset_index(drop=True)
    )

    # Create (id, df, model1_configs, model2_configs) tuple for each ID
    id_data_configs_list = []
    id_configs_dict = {}
    for id_, df_group in df_cleaned.groupby(id_col):

        # Determine model configurations
        model1_configs, model2_configs, cutoff_step = determine_model_configs(
            cfg, len(df_group)
        )
        id_data_config_dict = {
            "id": id_,
            "df": df_group,
            "cutoff_step": cutoff_step,
            "model1_configs": model1_configs,
            "model2_configs": model2_configs,
            "first_pred_date": first_pred_date,
        }

        id_configs_dict[id_] = {
            "model1_configs": model1_configs,
            "model2_configs": model2_configs,
        }
        id_data_configs_list.append(id_data_config_dict)

    # Save the id_configs_dict
    save_dict_to_json(id_configs_dict, os.path.join(work_dir, "id_configs_dict.json"))

    pred_df_list = []
    failed_ids = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_id = {
            executor.submit(
                fit_and_predict_single_id,
                id_data_config_dict,
                id_col,
                target_col,
                date_col,
                df_events_all, # custom-event
            ): id_data_config_dict["id"]
            for id_data_config_dict in id_data_configs_list
        }

        tqdm_out = TqdmToLoguru(logger)
        pbar = tqdm(
            total=len(future_to_id),
            desc="Processing IDs",
            file=tqdm_out,
        )
        completed_count = 0
        for future in as_completed(future_to_id):
            id_ = future_to_id[future]
            try:
                pred_df = future.result()
                if pred_df is not None:
                    pred_df_list.append(pred_df)
                else:
                    failed_ids.append(id_)
            except Exception as exc:
                logger.error(f"ID {id_} generated an exception: {exc}")
                failed_ids.append(id_)

            completed_count += 1

            if completed_count % log_interval == 0:
                pbar.update(log_interval)

        pbar.update(len(future_to_id) - pbar.n)
        pbar.close()

    if failed_ids:
        error_msg = f"The following IDs failed to fit: {failed_ids}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    pred_df = pd.concat(pred_df_list, ignore_index=True)
    logger.info(f"pred_df shape: {pred_df.shape}, {pred_df[id_col].unique().shape}")
    return pred_df


def fit_and_predict_single_id(
    id_data_config_dict,
    id_col="id",
    target_col="quantity_order",
    date_col="date",
    df_events_all=None, # custom-event
    clip_method=None,
    n_historic_predictions=False,
    minimal=True,
    progress=None,
):
    assert clip_method in [
        None,
        "zero",
        "min_id",
    ], "`clip_method` must be None or 'zero' or 'min_id'"

    # Parse data and model configurations from id_data_config_dict
    id_, df, cutoff_step, first_pred_date = parse_data_configs(id_data_config_dict)
    (
        model1_configs,
        model2_configs,
        freq_m1,
        freq_m2,
        use_events_m1,
        use_events_m2,
        fh_m1,
        fh_m2,
    ) = parse_model_configs(id_data_config_dict)

    try:
        logger.info(f"Fitting and predicting for ID: {id_}")

        # Copy the dataframe and use only necessary columns
        df = df[[id_col, date_col, target_col]].copy()
        # Rename columns to make it compatible with NeuralProphet
        df = df.rename(columns={id_col: "ID", date_col: "ds", target_col: "y"})

        # Set random seeds for reproducibility
        set_seeds(random_seed=4)

        # Initialize the models
        m1, df_w_events1, df_events_train1, df_events_untrain1 = fit_model_with_events( # custom-event
            df, model1_configs, freq_m1, minimal, progress, use_events_m1, df_events_all # custom-event
        )
        m2, df_w_events2, df_events_train2, df_events_untrain2 = fit_model_with_events( # custom-event
            df, model2_configs, freq_m2, minimal, progress, use_events_m2, df_events_all # custom-event
        )

        # Create future dataframe and generate forecast
        future1 = m1.make_future_dataframe(
            df_w_events1, events_df = df_events_train1, periods=fh_m1, n_historic_predictions=n_historic_predictions # custom-event
        )
        future2 = m2.make_future_dataframe(
            df_w_events2, events_df = df_events_train2, periods=fh_m2, n_historic_predictions=n_historic_predictions # custom-event
        )

        # Generate forecast
        forecast1, forecast2 = m1.predict(future1), m2.predict(future2)

        # Combine forecast1 and forecast2
        forecast_combine = combine_forecasts(
            forecast1,
            forecast2,
            cutoff_step,
            fh_m1,
            fh_m2,
            freq_m1,
            freq_m2,
            first_pred_date,
            common_freq=None,
        )

        if df_events_untrain2 is not None: # custom-event
            forecast_combine = evented_forecast(forecast_combine, df_events_untrain2) # custom-event
        forecast_combine.to_csv(f"forecast_combine_{id_}.csv",index=False) # custom-event

        # Apply clipping based on the clip_method
        if clip_method is not None:
            min_value = 0 if clip_method == "zero" else df["y"].min()
            forecast_combine["yhat"] = forecast_combine["yhat"].clip(lower=min_value)

        # Rename the target and date columns back to the original
        # and change the prediction column and store the results
        forecast_combine.rename(
            columns={"ID": id_col, "ds": date_col, "yhat1": "yhat"}, inplace=True
        )
        return forecast_combine[[id_col, date_col, "yhat"]]
    except Exception as e:
        logger.error(f"Error processing ID {id_}: {e}")
        return None


def fit_model_with_events(df, model_configs, freq, minimal, progress, use_events, df_events_all): # custom-event
    model = NeuralProphet(**model_configs)
    if use_events:
        model.add_country_holidays(country_name="US")
    df_w_events, df_events_train, df_events_untrain = df.copy(), None, None # custom-event
    if len(df_events_all) > 0: # custom-event
        df_events = df_events_all[df_events_all['ID'] == df['ID'].iloc[0]]  # custom-event
        if len(df_events) > 0: # custom-event
            df_events_train, df_events_untrain = decompose_df_events(df, df_events)  # custom-event
            model.add_events(df_events_train["event"].unique().tolist())  # custom-event
            df_w_events = model.create_df_with_events(df, df_events_train) # custom-event
    model.fit(df_w_events, freq=freq, num_workers=0, minimal=minimal, progress=progress) # custom-event
    return model, df_w_events, df_events_train, df_events_untrain # custom-event
