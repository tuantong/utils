import copy

import pandas as pd
from omegaconf import DictConfig

from forecasting.configs.logging_config import logger

WEEKLY_SEASONALITY_THRESHOLD = 52
MONTHLY_SEASONALITY_THRESHOLD = 12
DEFAULT_N_LAGS = 0
DEFAULT_N_FORECASTS = 1
TRAINER_CONFIG = {"accelerator": "cpu"}


def _set_model_config(
    base_config,
    forecast_horizon,
    n_changepoints,
    yearly_seasonality,
    n_lags=None,
    n_forecasts=None,
):
    growth = base_config.get("growth", "linear")
    n_changepoints = 0 if growth == "off" else n_changepoints
    config = {
        **base_config,
        "forecast_horizon": forecast_horizon,
        "n_changepoints": n_changepoints,
        "yearly_seasonality": yearly_seasonality,
        "trainer_config": base_config.get("trainer_config", TRAINER_CONFIG),
    }
    if n_lags is not None:
        config["n_lags"] = n_lags
    if n_forecasts is not None:
        config["n_forecasts"] = n_forecasts
    return config


def determine_model_configs(cfg: DictConfig, item_length: int):
    """
    Determine model configurations based on item length
    """
    cfg_copy = copy.deepcopy(cfg)
    # Dynamically select the base config based on item length
    cutoff_step = cfg_copy.model.cutoff_step
    # logger.info(f"Determining model configs for item length: {item_length}")

    # Retrieve base configs for both models
    base_config_model1 = cfg_copy.model.model1
    base_config_model2 = cfg_copy.model.model2

    # Get frequency from model configs
    freq_m1 = base_config_model1.freq
    freq_m2 = base_config_model2.freq

    # Determine changepoints based on frequency
    n_changepoints_m1 = (
        (item_length // 16) if freq_m1 == "W-MON" else (item_length // 4)
    )
    n_changepoints_m2 = 0  # Disable changepoints for model 2

    # Determine n_lags and n_forecasts with fallback defaults
    n_lags_m1 = cfg_copy.model.model1.get("n_lags", DEFAULT_N_LAGS)
    n_forecasts_m1 = cfg_copy.model.model1.get("n_forecasts", DEFAULT_N_FORECASTS)
    n_lags_m2 = cfg_copy.model.model2.get("n_lags", DEFAULT_N_LAGS)
    n_forecasts_m2 = cfg_copy.model.model2.get("n_forecasts", DEFAULT_N_FORECASTS)

    # Determine yearly seasonality based on item_length and frequency
    yearly_seasonality_m1 = (
        item_length >= WEEKLY_SEASONALITY_THRESHOLD
        if freq_m1 == "W-MON"
        else item_length >= MONTHLY_SEASONALITY_THRESHOLD
    )
    yearly_seasonality_m2 = (
        item_length >= WEEKLY_SEASONALITY_THRESHOLD
        if freq_m2 == "W-MON"
        else item_length >= MONTHLY_SEASONALITY_THRESHOLD
    )

    # Adjust model1 config based on item length
    if item_length < (n_lags_m1 + n_forecasts_m1):
        logger.warning(
            f"Item has length less than {n_lags_m1 + n_forecasts_m1}, setting n_lags_m1 to 0 and n_forecasts_m1 to 1"
        )
        n_lags_m1 = 0
        n_forecasts_m1 = 1
        logger.info(
            "Setting growth to 'linear' and AR regularization to 0 due to insufficient item length."
        )
        cfg_copy.model.model1["growth"] = "linear"
        cfg_copy.model.model1["ar_reg"] = 0

    # Adjust model2 config based on item length
    if item_length < (n_lags_m2 + n_forecasts_m2):
        logger.warning(
            f"Item has length less than {n_lags_m2 + n_forecasts_m2}, setting n_lags_m2 to 0 and n_forecasts_m2 to 1"
        )
        n_lags_m2 = 0
        n_forecasts_m2 = 1

    # Get models forecast horizon
    forecast_horizon_m1 = base_config_model1.forecast_horizon
    forecast_horizon_m2 = base_config_model2.forecast_horizon

    model1_configs = _set_model_config(
        base_config_model1,
        forecast_horizon=forecast_horizon_m1,
        n_changepoints=n_changepoints_m1,
        yearly_seasonality=yearly_seasonality_m1,
        n_lags=n_lags_m1,
        n_forecasts=n_forecasts_m1,
    )
    model2_configs = _set_model_config(
        base_config_model2,
        forecast_horizon=forecast_horizon_m2,
        n_changepoints=n_changepoints_m2,
        yearly_seasonality=yearly_seasonality_m2,
        n_lags=n_lags_m2,
        n_forecasts=n_forecasts_m2,
    )

    return model1_configs, model2_configs, cutoff_step


def is_last_month_complete(weekly_df: pd.DataFrame, date_col: str) -> bool:
    """Check if the last labeled month is complete based on 'W-MON' frequency."""
    # Get the most recent labeled week
    latest_date = weekly_df[date_col].max()

    # Check if the latest date is in the same month as the previous month’s end
    return latest_date.month != (latest_date + pd.DateOffset(days=7)).month


def parse_model_configs(id_data_config_dict):
    """Parse model configurations from id_data_config_dict"""
    # Model configs, frequency and use_events extraction
    # Ensure configs are copied
    model1_configs = id_data_config_dict.get("model1_configs", {}).copy()
    model2_configs = id_data_config_dict.get("model2_configs", {}).copy()
    freq_m1 = model1_configs.pop("freq", None)
    freq_m2 = model2_configs.pop("freq", None)

    # Extract use_events for each model
    use_events_m1 = model1_configs.pop("use_events", False)
    use_events_m2 = model2_configs.pop("use_events", False)

    # Extract forecast horizon for each model
    forecast_horizon_m1 = model1_configs.pop("forecast_horizon", None)
    forecast_horizon_m2 = model2_configs.pop("forecast_horizon", None)

    return (
        model1_configs,
        model2_configs,
        freq_m1,
        freq_m2,
        use_events_m1,
        use_events_m2,
        forecast_horizon_m1,
        forecast_horizon_m2,
    )


def parse_data_configs(id_data_config_dict):
    """Parse data configurations from id_data_config_dict"""
    id_ = id_data_config_dict["id"]
    df = id_data_config_dict["df"]
    cutoff_step = id_data_config_dict["cutoff_step"]
    first_pred_date = id_data_config_dict["first_pred_date"]

    return id_, df, cutoff_step, first_pred_date


def validate_config(cfg: DictConfig):
    """Validate data and model configs."""
    assert cfg.data is not None, "Data field must be provided in the config."
    assert cfg.model is not None, "Model field must be provided in the config."
    assert (
        cfg.model.model1 is not None and cfg.model.model2 is not None
    ), "Both model1 and model2 must be provided in the config."
    assert (
        cfg.model.model1.freq is not None and cfg.model.model2.freq is not None
    ), "Both model1 and model2 must have a 'freq' field in the config."
    assert (
        cfg.model.model1.forecast_horizon is not None
        and cfg.model.model2.forecast_horizon is not None
    ), "Both model1 and model2 must have a 'forecast_horizon' field in the config."
    assert (
        cfg.model.cutoff_step is not None
    ), "'cutoff_step' must be provided in the model's config."

    assert (
        cfg.model.model1.freq == cfg.model.model2.freq
    ), "Both models must have the same frequency, different frequencies are not supported yet."
    assert cfg.model.model1.freq in ["W-MON", "M"] and cfg.model.model2.freq in [
        "W-MON",
        "M",
    ], "Only 'W-MON' and 'M' frequencies are supported."

    assert (
        cfg.model.cutoff_step <= cfg.model.model1.forecast_horizon
    ), "Model cutoff_step must be less than or equal to model1's forecast horizon."
