from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


def evaluation_temp_speed(actual_dataframe, forecast_dataframe, model_name, for_temp, for_speed, temp, speed):
    """
        Evaluates the performance of temperature and wind speed forecasts using various metrics.

        Params:
            actual_dataframe (pd.DataFrame): The DataFrame containing actual observed values.
            forecast_dataframe (pd.DataFrame): The DataFrame containing forecasted values from the model.
            model_name (str): The name of the model being evaluated.
            for_temp (str): The column name for forecasted temperature values.
            for_speed (str): The column name for forecasted wind speed values.
            temp (str): The column name for actual temperature values.
            speed (str): The column name for actual wind speed values.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics for temperature and wind speed,
                          including MAE, MSE, RMSE, and R² values for both parameters.
        """
    # Create a dictionary to store metrics
    metrics = {
        f'Metrics for {model_name}': ['Temperature MAE', 'Temperature MSE', 'Temperature RMSE', 'Temperature R2',
                                      'Wind Speed MAE', 'Wind Speed MSE', 'Wind Speed RMSE', 'Wind Speed R2'],
        'Value': [
            mean_absolute_error(forecast_dataframe[for_temp], actual_dataframe[temp]),
            mean_squared_error(forecast_dataframe[for_temp], actual_dataframe[temp]),
            np.sqrt(mean_squared_error(forecast_dataframe[for_temp], actual_dataframe[temp])),
            r2_score(forecast_dataframe[for_temp], actual_dataframe[temp]),
            mean_absolute_error(forecast_dataframe[for_speed], actual_dataframe[speed]),
            mean_squared_error(forecast_dataframe[for_speed], actual_dataframe[speed]),
            np.sqrt(mean_squared_error(forecast_dataframe[for_speed], actual_dataframe[speed])),
            r2_score(forecast_dataframe[for_speed], actual_dataframe[speed])
        ]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index(f'Metrics for {model_name}')
    return metrics_df


def evaluation_direction(gfs_data, ifs_data, iconeu_data, gfs_angular_diff, ifs_angular_diff, iconeu_angular_diff):
    """
        Evaluates the performance of wind direction forecasts using various metrics.

        Params:
            gfs_data (pd.DataFrame): The DataFrame containing GFS forecasted data, including angular differences.
            ifs_data (pd.DataFrame): The DataFrame containing IFS forecasted data, including angular differences.
            iconeu_data (pd.DataFrame): The DataFrame containing ICONEU forecasted data, including angular differences.
            gfs_angular_diff (str): The column name for GFS angular differences.
            ifs_angular_diff (str): The column name for IFS angular differences.
            iconeu_angular_diff (str): The column name for ICONEU angular differences.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics for wind direction forecasts,
                          including Mean Absolute Error (MAE), Median Absolute Error, and Root Mean Absolute Square
                          Error (RMASE) for each model.
        """
    metrics = {
        'Wind Direction Metrics': ['MAE GFS', 'MAE IFS', 'MAE ICONEU', 'Median Abs. Err. GFS',
                                   'Median Abs. Err. IFS', 'Median Abs. Err. ICONEU', 'RMSE GFS',
                                   'RMSE IFS', 'RMSE ICONEU'],
        'Value': [
            np.mean(np.abs(gfs_data[gfs_angular_diff])),
            np.mean(np.abs(ifs_data[ifs_angular_diff])),
            np.mean(np.abs(iconeu_data[iconeu_angular_diff])),
            np.median(np.abs(gfs_data[gfs_angular_diff])),
            np.median(np.abs(ifs_data[ifs_angular_diff])),
            np.median(np.abs(iconeu_data[iconeu_angular_diff])),
            np.sqrt(np.mean(np.square(gfs_data[gfs_angular_diff]))),
            np.sqrt(np.mean(np.square(ifs_data[ifs_angular_diff]))),
            np.sqrt(np.mean(np.square(iconeu_data[iconeu_angular_diff])))
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.set_index('Wind Direction Metrics')
    return metrics_df
