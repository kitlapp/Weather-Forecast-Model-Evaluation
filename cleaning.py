import numpy as np
import pandas as pd  # For data manipulation and analysis.
from tabulate import tabulate


def tidy_the_outputs(dataframe, col_names):
    """
    Renames the columns of the given DataFrame according to the provided mapping.

    Params:
        dataframe (pd.DataFrame): The DataFrame whose columns need to be renamed.
        col_names (dict): A dictionary where keys are the current column names, and values are the new column names.

    Returns:
        pd.DataFrame: A DataFrame with renamed columns.
    """
    return dataframe.rename(columns=col_names)


def fix_datetime(dataframe):
    """
        Converts the 'Date' column of the given DataFrame to datetime format.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing a 'Date' column as strings or other formats.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with the 'Date' column converted to datetime.
        """
    df = dataframe.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def show_info(dataframe, dataframe_info):
    """
        Prints basic information about the given DataFrame, including its head, tail, and summary info.

        Params:
            dataframe (pd.DataFrame): The DataFrame to display information for.
            dataframe_info (str): A custom description or label for the DataFrame that will be printed.

        Returns:
            None: This function only prints the DataFrame's information.
        """
    print(f'{dataframe_info}\n')
    # Pretty print DataFrames
    print(tabulate(dataframe.head(), headers='keys', tablefmt='simple'))
    print(tabulate(dataframe.tail(), headers='keys', tablefmt='simple'))
    print(dataframe.info())


def handle_nan(dataframe, temp, speed, direction_meteo):
    """
        Handles missing values in the specified DataFrame by interpolating values for temperature, wind speed,
        and wind direction.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing the data with missing values.
            temp (str): The column name for temperature data to interpolate using linear interpolation.
            speed (str): The column name for wind speed data to interpolate using linear interpolation.
            direction_meteo (str): The column name for wind direction data to interpolate using nearest
                                   interpolation followed by forward fill.

        Returns:
            pd.DataFrame: A DataFrame with missing values handled through interpolation.
        """
    df = dataframe.copy()
    df[temp] = df[temp].interpolate(method='linear')
    df[speed] = df[speed].interpolate(method='linear')
    df[direction_meteo] = df[direction_meteo].interpolate(method='nearest').ffill()
    return df


def add_nan_col(dataframe, nan_column):
    """
        Adds a new column to the DataFrame indicating the presence of any NaN values in each row.

        Params:
            dataframe (pd.DataFrame): The DataFrame to which the new column will be added.
            nan_column (str): The name of the new column that will indicate rows with NaN values
                              (1 for NaN present, 0 for no NaN).

        Returns:
            pd.DataFrame: A DataFrame with the new column added, showing the presence of NaN values.
        """
    df = dataframe.copy()
    df[nan_column] = df.isna().any(axis=1)
    df[nan_column] = df[nan_column].astype(int)
    return df


def group_nan_by_day_hour(dataframe, day, hour):
    """
        Groups the DataFrame by day and hour to count the number of NaN values.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing a 'Date' column and a 'Includes NaN' column.
            day (str): The name of the new column to store the day extracted from the 'Date' column.
            hour (str): The name of the new column to store the hour extracted from the 'Date' column.

        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: A DataFrame with counts of NaN values grouped by day and hour.
                - pd.DataFrame: A DataFrame with total counts of NaN values grouped by day.
                - pd.DataFrame: A DataFrame with total counts of NaN values grouped by hour.
        """
    df = dataframe.copy()
    df[day] = df['Date'].dt.date
    df[hour] = df['Date'].dt.hour
    df = df.groupby([day, hour])['Includes NaN'].sum().reset_index()
    df = df.rename(columns=({'Includes NaN': 'Number of NaN'}))
    df_2 = df.groupby(day)['Number of NaN'].sum().reset_index()
    df_3 = df.groupby(hour)['Number of NaN'].sum().reset_index()
    return df, df_2, df_3


def keep_nan_add_time_col(dataframe, day, hour, minute, second):
    """
        Filters the DataFrame to keep only rows with NaN values and adds separate time columns.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing a 'Date' column and an 'Includes NaN' column.
            day (str): The name of the new column to store the day extracted from the 'Date' column.
            hour (str): The name of the new column to store the hour extracted from the 'Date' column.
            minute (str): The name of the new column to store the minute extracted from the 'Date' column.
            second (str): The name of the new column to store the second extracted from the 'Date' column.

        Returns:
            pd.DataFrame: A DataFrame with only the rows containing NaN values and additional columns for
                          day, hour, minute, second, and total time in seconds.
        """
    df = dataframe.copy()
    df = df[df['Includes NaN'] == 1]
    df = df['Date'].reset_index()
    df[day] = df['Date'].dt.day
    df[hour] = df['Date'].dt.hour
    df[minute] = df['Date'].dt.minute
    df[second] = df['Date'].dt.second
    df['Time in Secs'] = ((df[day] - 1) * 24 * 3600 + df[hour] * 60 * 60 + df[minute] * 60 + df[second])
    df = df.drop(columns='index')
    df.reset_index(drop=True)
    return df


def resample_hourly(dataframe, temp, speed, u, v):
    """
        Resamples the DataFrame to hourly frequency, aggregating temperature, wind speed, and wind components.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing a 'Date' column and the specified meteorological data.
            temp (str): The column name for temperature data to keep the first value of each hour.
            speed (str): The column name for wind speed data to average over each hour.
            u (str): The column name for the U component of wind to average over each hour.
            v (str): The column name for the V component of wind to average over each hour.

        Returns:
            pd.DataFrame: A resampled DataFrame with hourly frequency and aggregated values for temperature,
                          wind speed, and wind components.
        """
    df = dataframe.set_index('Date').resample('h').agg({
        temp: 'first',  # Keep the temperature value at the start of each hour
        speed: 'mean',  # Average hourly wind
        u: 'mean',
        v: 'mean'
    }).reset_index()
    return df


def adjust_forecast_data(dataframe, wind_columns_to_be_adjusted):
    """
        Adjusts the forecast DataFrame by shifting specified wind columns by one time step.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing forecast data with wind columns.
            wind_columns_to_be_adjusted (list of str): A list of column names for wind data that
                                                        need to be shifted.

        Returns:
            pd.DataFrame: A DataFrame with the specified wind columns shifted by one time step,
                          excluding the last row to maintain alignment.
        """
    df = dataframe.copy()
    df[wind_columns_to_be_adjusted] = df[wind_columns_to_be_adjusted].shift(-1)
    df = df[:-1]
    return df


def convert_comp_to_math_and_meteo_angles(dataframe, math_angle, meteo_angle, y, x):
    """
        Converts Cartesian wind components to mathematical and meteorological angles.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing the wind component data.
            math_angle (str): The name of the column to store the mathematical angle in degrees.
            meteo_angle (str): The name of the column to store the meteorological angle in degrees.
            y (str): The name of the column representing the Y component (vertical) of the wind.
            x (str): The name of the column representing the X component (horizontal) of the wind.

        Returns:
            pd.DataFrame: A DataFrame with new columns for the mathematical and meteorological angles
                          calculated from the wind components.
        """
    df = dataframe.copy()
    df[math_angle] = (np.rad2deg(np.arctan2(df[y], df[x])) + 360) % 360
    df[meteo_angle] = ((270 - df[math_angle]) + 360) % 360
    return df


def create_wind_direction_df(dataframe, gfs_data, ifs_data, iconeu_data):
    """
        Creates a DataFrame for wind direction by adding meteorological angles from various forecast datasets.

        Params:
            dataframe (pd.DataFrame): The original DataFrame containing wind and temperature data.
            gfs_data (pd.DataFrame): The DataFrame containing GFS meteorological angle data.
            ifs_data (pd.DataFrame): The DataFrame containing IFS meteorological angle data.
            iconeu_data (pd.DataFrame): The DataFrame containing ICONEU meteorological angle data.

        Returns:
            pd.DataFrame: A DataFrame with wind direction angles from GFS, IFS, and ICONEU, excluding
                          specified columns from the original DataFrame.
        """
    df = dataframe.copy()
    df = df.drop(columns=['Temperature', 'Wind Speed', 'Zonal X Comp. (u)', 'Meridional Y Comp. (v)', 'Math Angle'])
    df['GFS Meteo Angle'] = gfs_data['GFS Dir Meteo']
    df['IFS Meteo Angle'] = ifs_data['IFS Dir Meteo']
    df['ICONEU Meteo Angle'] = iconeu_data['ICONEU Dir Meteo']
    return df


def angular_difference(angle1, angle2):
    """
        Calculates the minimum angular difference between two angles, accounting for circularity.

        Params:
            angle1 (float): The first angle in degrees.
            angle2 (float): The second angle in degrees.

        Returns:
            float: The minimum angular difference between the two angles, in degrees.
        """
    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, 360 - diff)  # Adjust for circular nature
