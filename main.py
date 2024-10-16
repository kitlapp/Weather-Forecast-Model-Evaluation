# ======================================================================================================================
# *** IMPORT LIBRARIES AND READ FILES *** IMPORT LIBRARIES AND READ FILES *** IMPORT LIBRARIES AND READ FILES ***
# ======================================================================================================================

import numpy as np  # For numerical operations and arrays.
import pandas as pd  # For data manipulation and analysis.
import matplotlib.pyplot as plt  # For basic plotting.
import seaborn as sns  # For enhanced plotting.
from cleaning import (tidy_the_outputs, fix_datetime, show_info, handle_nan, add_nan_col, group_nan_by_day_hour,
                      keep_nan_add_time_col, resample_hourly, adjust_forecast_data,  # Custom functions for cleaning.
                      convert_comp_to_math_and_meteo_angles, create_wind_direction_df, angular_difference)
from dictionaries import STATION_DATA_COL_NAMES, FORECAST_DATA_COL_NAMES  # Custom dicts for renaming columns.
from visualizations import create_vertical_barplot  # Custom functions for visualizations.
from evaluation import evaluation_temp_speed, evaluation_direction  # Custom functions for evaluation.

pd.options.display.max_columns = 20

# Read the parquet files to dfs:
station_raw = pd.read_parquet('station_data.parquet')
forecast_raw = pd.read_parquet('forecast_data.parquet')

# ======================================================================================================================
# *** DATA PREPROCESSING *** DATA PREPROCESSING *** DATA PREPROCESSING *** DATA PREPROCESSING *** DATA PREPROCESSING ***
# ======================================================================================================================
# *** Basic Data Preprocessing ***
# ======================================================================================================================

# Custom function to rename columns at both datasets:
station_corrected_names = tidy_the_outputs(dataframe=station_raw, col_names=STATION_DATA_COL_NAMES)
forecast_corrected_names = tidy_the_outputs(dataframe=forecast_raw, col_names=FORECAST_DATA_COL_NAMES)

# Custom function to set 'Date' to be of a datetime Dtype:
station_fix_dtype = fix_datetime(dataframe=station_corrected_names)
forecast_fix_dtype = fix_datetime(dataframe=forecast_corrected_names)

# ======================================================================================================================
# *** Weather Station Data (Actual Data) Preprocessing ***
# ======================================================================================================================

# Custom function to fill the NaN:
station_nan = handle_nan(dataframe=station_fix_dtype,
                         temp='Temperature',
                         speed='Wind Speed',
                         direction_meteo='Direction Meteo')

# Custom function to add the new col 'Includes NaN' astype(int) into a df:
station_plus_nan_col = add_nan_col(dataframe=station_fix_dtype, nan_column='Includes NaN')

# Custom function to return a tuple with NaN by day and hour, NaN by day and NaN by hour:
station_nan_day_hour, station_nan_day, station_nan_hour = group_nan_by_day_hour(dataframe=station_plus_nan_col,
                                                                                day='Day',
                                                                                hour='Hour')

# Custom function to keep only the NaN rows of a df and add a new col with the timestamps in seconds (Time in Secs):
station_nan_only_plus_time_col = keep_nan_add_time_col(dataframe=station_plus_nan_col,
                                                       day='Day',
                                                       hour='Hour',
                                                       minute='Minute',
                                                       second='Second')
# *** !!! THE 4 DFS ABOVE ARE IDEAL FOR MISSING VALUES EXPLORATION AND VISUALIZATIONS !!! ***

# Add a new column including the mathematical angles in degrees:
station_plus_math = station_nan.copy()
station_plus_math['Direction Math'] = ((270 - station_plus_math['Direction Meteo']) + 360) % 360

# Add a new column including the mathematical angles in radians:
station_plus_rad = station_plus_math.copy()
station_plus_rad['Direction Math (rad)'] = np.radians(station_plus_rad['Direction Math'])

# **********!!! IMPORTANT CODE SCRIPT BELOW !!!**********
# Add new columns including both the zonal (u) and meridional (v) components of the math angles in radians:
station_plus_u_v = station_plus_rad.copy()
station_plus_u_v['Zonal X Comp. (u)'] = np.cos(station_plus_u_v['Direction Math (rad)'])
station_plus_u_v['Meridional Y Comp. (v)'] = np.sin(station_plus_u_v['Direction Math (rad)'])
# *** Angles can't be resampled per hour by calculating their mean: BUT ZONAL AND MERIDIONAL COMPONENTS CAN! ***
# ********** THE ABOVE DATAFRAME CAN BE EFFECTIVELY RESAMPLED **********

# Custom function returning hourly resampled values of a df.
station_resampled = resample_hourly(dataframe=station_plus_u_v,
                                    temp='Temperature',
                                    speed='Wind Speed',
                                    u='Zonal X Comp. (u)',
                                    v='Meridional Y Comp. (v)')

# Custom function to convert the wind components after resampling back to angles:
station_plus_angles = convert_comp_to_math_and_meteo_angles(dataframe=station_resampled,
                                                            math_angle='Math Angle',
                                                            meteo_angle='Meteo Angle',
                                                            y='Meridional Y Comp. (v)',
                                                            x='Zonal X Comp. (u)')

# **********!!! IMPORTANT CODE SCRIPT BELOW !!!**********
# Create the final weather station (actual) datasets:
gfs_actual = station_plus_angles.copy()  # Actual data for GFS from 2024-06-01 00:00:00 to 2024-06-07 23:00:00
ifs_actual = station_plus_angles.iloc[:144]  # Actual data for IFS from 2024-06-01 00:00:00 to 2024-06-06 23:00:00
iconeu_actual = station_plus_angles.iloc[:120]  # Actual data for ICONEU from 2024-06-01 00:00:00 to 2024-06-05 23:00:00
# *** THIS IS THE FINAL DATAFRAMES FOR ACTUAL DATA: ACTUAL DATA READY FOR COMPARISONS ***

# Custom function to explore the actual data dfs:
show_info(dataframe=gfs_actual, dataframe_info='THIS DF CONTAINS ACTUAL DATA TO BE COMPARED WITH GFS FORECASTS:')
show_info(dataframe=ifs_actual, dataframe_info='THIS DF CONTAINS ACTUAL DATA TO BE COMPARED WITH IFS FORECASTS:')
show_info(dataframe=iconeu_actual, dataframe_info='THIS DF CONTAINS ACTUAL DATA TO BE COMPARED WITH ICONEU FORECASTS:')

# ======================================================================================================================
# *** Forecast Data Preprocessing ***
# ======================================================================================================================

# Call a function shifting the forecast wind cols to the right position:
forecast_adjusted = adjust_forecast_data(dataframe=forecast_fix_dtype,
                                         wind_columns_to_be_adjusted=['GFS Wind Speed', 'IFS Wind Speed',
                                                                      'ICONEU Wind Speed', 'GFS Dir Meteo',
                                                                      'IFS Dir Meteo', 'ICONEU Dir Meteo'])

# Since the wind cols shifted to -1, temperatures of the last shifted row should be set to NaN:
forecast_adjusted.loc[144, 'IFS Temp'] = np.nan  # Row Index 144: 2024-06-07 00:00:00
forecast_adjusted.loc[120, 'ICONEU Temp'] = np.nan  # Row Index 120: 2024-06-06 00:00:00

# Filter out rows with missing values for IFS and ICONEU:
forecast_ifs_non_missing = forecast_adjusted[forecast_adjusted['IFS Temp'].notna()]
forecast_iconeu_non_missing = forecast_adjusted[forecast_adjusted['ICONEU Temp'].notna()]

# **********!!! IMPORTANT CODE SCRIPT BELOW !!!**********
# Create the final GFS, IFS, ICONEU (forecast) datasets:
gfs_forecast = forecast_adjusted.copy()  # GFS forecast from 2024-06-01 00:00:00 to 2024-06-07 23:00:00
ifs_forecast = forecast_ifs_non_missing.copy()  # IFS Forecast from 2024-06-01 00:00:00 to 2024-06-06 23:00:00
iconeu_forecast = forecast_iconeu_non_missing.copy()  # ICONEU Forecast from 2024-06-01 00:00:00 to 2024-06-05 23:00:00
# *** THIS IS THE FINAL DATAFRAMES FOR FORECAST DATA: FORECAST DATA READY FOR COMPARISONS ***

# Custom function to explore the forecast data dfs:
show_info(dataframe=gfs_forecast, dataframe_info='THIS DF CONTAINS GFS FORECASTS:')
show_info(dataframe=ifs_forecast, dataframe_info='THIS DF CONTAINS IFS FORECASTS:')
show_info(dataframe=iconeu_forecast, dataframe_info='THIS DF CONTAINS ICONEU FORECASTS:')

# ======================================================================================================================
# *** Wind Direction Data Preprocessing ***
# ======================================================================================================================

# **********!!! Wind directions ir preprocessed separately as it includes angles !!!**********

# Call a function to create a new df with the appropriate columns for wind direction analysis:
wind_direction = create_wind_direction_df(
    dataframe=station_plus_angles,
    gfs_data=gfs_forecast,
    ifs_data=ifs_forecast,
    iconeu_data=iconeu_forecast
)

# Call a function to properly calculate the angular difference:
wind_direction['GFS Angular Diff.'] = angular_difference(angle1=wind_direction['Meteo Angle'],
                                                         angle2=wind_direction['GFS Meteo Angle'])
wind_direction['IFS Angular Diff.'] = angular_difference(angle1=wind_direction['Meteo Angle'],
                                                         angle2=wind_direction['IFS Meteo Angle'])
wind_direction['ICONEU Angular Diff.'] = angular_difference(angle1=wind_direction['Meteo Angle'],
                                                            angle2=wind_direction['ICONEU Meteo Angle'])

# **********!!! IMPORTANT CODE SCRIPT BELOW !!!**********
gfs_wind_direction = wind_direction.copy()  # from 2024-06-01 00:00:00 to 2024-06-07 23:00:00
ifs_wind_direction = wind_direction.iloc[:144]  # from 2024-06-01 00:00:00 to 2024-06-06 23:00:00
iconeu_wind_direction = wind_direction.iloc[:120]  # from 2024-06-01 00:00:00 to 2024-06-05 23:00:00
# *** THIS IS THE FINAL DATAFRAMES FOR WIND DIRECTION DATA: DATA READY FOR EVALUATION ***

# Custom function to explore the wind direction dfs:
show_info(dataframe=gfs_wind_direction, dataframe_info='THIS DF CONTAINS WIND DIRECTION DATA TO EVALUATE GFS:')
show_info(dataframe=ifs_wind_direction, dataframe_info='THIS DF CONTAINS WIND DIRECTION DATA TO EVALUATE IFS:')
show_info(dataframe=iconeu_wind_direction, dataframe_info='THIS DF CONTAINS WIND DIRECTION DATA TO EVALUATE ICONEU:')

# ======================================================================================================================
# *** EXPLORATORY DATA ANALYSIS *** EXPLORATORY DATA ANALYSIS *** EXPLORATORY DATA ANALYSIS ***
# ======================================================================================================================
# *** Basic Exploratory Data Analysis ***
# ======================================================================================================================

# Explore common statistics of the original datasets:
print('Actual Raw Data Statistics:\n', station_raw.describe().transpose().iloc[:-1])
print('Forecast Raw Data Statistics:\n', forecast_raw.describe().transpose().iloc[:-3])

# ======================================================================================================================
# *** Exploratory Data Analysis on Missing Values ***
# ======================================================================================================================

# Show the missing values per day:
create_vertical_barplot(dataframe=station_nan_day,
                        x_axis='Day',
                        y_axis='Number of NaN',
                        y_lim=1100,
                        title='Total Missing Values per Day',
                        x_label='Date',
                        x_ticks_rot=20,
                        y_label='Number of Missing Values',
                        y_ticks_rot=0)

# Show the missing values per hour:
create_vertical_barplot(dataframe=station_nan_hour,
                        x_axis='Hour',
                        y_axis='Number of NaN',
                        y_lim=225,
                        title='Total Missing Values per Hour from 01 το 07/06/2024',
                        x_label='Hour',
                        x_ticks_rot=0,
                        y_label='Number of Missing Values',
                        y_ticks_rot=0)

# Create heatmap data for missing values per day and hour:
heatmap_data = station_nan_day_hour.pivot(index='Day', columns='Hour', values='Number of NaN')

# Show the heatmap above:
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='d', linewidths=0.5)
plt.title('Total Missing Values per Date and Hour')
plt.xlabel('Hour')
plt.ylabel('Date')
plt.yticks(rotation=20)
plt.show()

# Show a time series with the evolution of missing values over time:
plt.figure(figsize=(10, 6))
plt.plot(station_nan_only_plus_time_col['Time in Secs'])
plt.title('Evolution of Missing Values over Time')
plt.xlabel('Number of Missing Values')
plt.xlim(0, 3000)
plt.xticks(np.arange(0, 3001, 250), rotation=0)
plt.ylabel('Time in Seconds')
plt.ylim(0, 650_000)
plt.yticks(np.arange(0, 650_001, 50000), rotation=20)
plt.grid(True)
plt.show()

# ======================================================================================================================
# *** Exploratory Data Analysis on Temperature ***
# ======================================================================================================================

# Show a time series with the actual and the forecast temperature:
plt.figure(figsize=(10, 6))
plt.plot(station_resampled['Date'], station_resampled['Temperature'], label='Actual Temperature')
plt.plot(gfs_forecast['Date'], gfs_forecast['GFS Temp'], label='GFS Temperature', color='orange')
plt.plot(ifs_forecast['Date'], ifs_forecast['IFS Temp'], label='IFS Temperature', color='green')
plt.plot(iconeu_forecast['Date'], iconeu_forecast['ICONEU Temp'], label='ICONEU Temperature', color='red')
plt.xlabel('Date')
plt.xticks(rotation=20)
plt.ylabel('Temperature (°C)')
plt.title('Actual VS Predicted Temperature over Time')
plt.legend()
plt.grid(True)
plt.show()

# Create heatmap data for temperature values per day and hour:
heatmap_temp_day_hour = station_resampled.copy()
heatmap_temp_day_hour['Hour'] = heatmap_temp_day_hour['Date'].dt.hour
heatmap_temp_day_hour['Day_Date'] = heatmap_temp_day_hour['Date'].dt.date

# Show the heatmap above:
temp_vs_day_hour_pivot = heatmap_temp_day_hour.pivot_table(values='Temperature', index='Day_Date', columns='Hour')
plt.figure(figsize=(10, 6))
sns.heatmap(temp_vs_day_hour_pivot, cmap='coolwarm', annot=True, linewidths=0.5)
plt.title('Actual Temperature per Date and Hour')
plt.xlabel('Hour')
plt.ylabel('Date')
plt.yticks(rotation=20)
plt.show()

# ======================================================================================================================
# *** Exploratory Data Analysis on Wind Speed ***
# ======================================================================================================================

# Create a df including the speeds which lesser to 1 mph:
explore_calm_wind_data = station_raw[station_raw['wind_speed'] < 0.447]
print('Explore the Calm Wind DataFrame:\n')
explore_calm_wind_data.info()

# Show the wind speed variations by date:
plt.figure(figsize=(10, 6))
sns.boxplot(x=station_resampled['Date'].dt.date, y=station_resampled['Wind Speed'])
plt.xticks(rotation=20)
plt.xlabel('Date')
plt.ylabel('Wind Speed (m/s)')
plt.title('Actual Wind Speed Variation by Date')
plt.grid(True)
plt.show()

# Show a time series with the actual and the forecast wind speeds:
plt.figure(figsize=(10, 6))
plt.plot(station_resampled['Date'], station_resampled['Wind Speed'], label='Actual Wind Speed')
plt.plot(gfs_forecast['Date'], gfs_forecast['GFS Wind Speed'], label='GFS Wind Speed', color='orange')
plt.plot(ifs_forecast['Date'], ifs_forecast['IFS Wind Speed'], label='IFS Wind Speed', color='green')
plt.plot(iconeu_forecast['Date'], iconeu_forecast['ICONEU Wind Speed'], label='ICONEU Wind Speed', color='red')
plt.xlabel('Date')
plt.xticks(rotation=20)
plt.ylabel('Wind Speed (m/s)')
plt.title('Actual VS Predicted Wind Speed over Time')
plt.grid(True)
plt.legend()

# ======================================================================================================================
# *** Exploratory Data Analysis on Wind Direction ***
# ======================================================================================================================

# Show circular histograms for wind direction in the same figure:

# Create a figure with subplots:
fig, axs = plt.subplots(1, 3, figsize=(10, 18), subplot_kw={'polar': True})
angles = np.linspace(0, 2 * np.pi, 9)

# Create circular histogram for actual data vs GFS forecasts:
axs[0].hist(np.deg2rad(gfs_actual['Meteo Angle']), bins=36, alpha=0.5, edgecolor='black', linewidth=0.7, label='Actual')
axs[0].hist(np.deg2rad(gfs_forecast['GFS Dir Meteo']), bins=36, alpha=0.5, edgecolor='black', color='orange',
            linewidth=0.7, label='GFS')
axs[0].set_title('Wind Direction - Actual vs GFS', va='bottom')
axs[0].set_theta_direction(-1)
axs[0].set_theta_zero_location('N')
axs[0].set_xticks(np.linspace(0, 2 * np.pi, 9))
axs[0].set_xticks(angles[:-1])
axs[0].set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
axs[0].legend(loc='upper right', bbox_to_anchor=(0.35, 0.04))

# Create circular histogram for actual data vs IFS forecasts:
axs[1].hist(np.deg2rad(ifs_actual['Meteo Angle']), bins=36, alpha=0.5, edgecolor='black', linewidth=0.7, label='Actual')
axs[1].hist(np.deg2rad(ifs_forecast['IFS Dir Meteo']), bins=36, alpha=0.5, edgecolor='black', color='green',
            linewidth=0.7, label='IFS')
axs[1].set_title('Wind Direction - Actual vs IFS', va='bottom')
axs[1].set_theta_direction(-1)
axs[1].set_theta_zero_location('N')
axs[1].set_xticks(np.linspace(0, 2 * np.pi, 9))
axs[1].set_xticks(angles[:-1])
axs[1].set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
axs[1].legend(loc='upper right', bbox_to_anchor=(0.35, 0.04))

# Create circular histogram for actual data vs ICONEU forecasts:
axs[2].hist(np.deg2rad(iconeu_actual['Meteo Angle']), bins=36, alpha=0.5, edgecolor='black', linewidth=0.7,
            label='Actual')
axs[2].hist(np.deg2rad(ifs_forecast['ICONEU Dir Meteo']), bins=36, alpha=0.5, edgecolor='black', color='red',
            linewidth=0.7, label='ICONEU')
axs[2].set_title('Wind Direction - Actual vs ICONEU', va='bottom')
axs[2].set_theta_direction(-1)
axs[2].set_theta_zero_location('N')
axs[2].set_xticks(np.linspace(0, 2 * np.pi, 9))
axs[2].set_xticks(angles[:-1])
axs[2].set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
axs[2].legend(loc='upper right', bbox_to_anchor=(0.35, 0.04))

plt.show()

# ======================================================================================================================
# *** EVALUATION *** EVALUATION *** EVALUATION *** EVALUATION *** EVALUATION *** EVALUATION *** EVALUATION ***
# ======================================================================================================================
# *** Temperature and Wind Speed Evaluation ***
# ======================================================================================================================

# Call a function to evaluate temperature and wind speed using 4 different metrics (MAE, MSE, RMSE, R2):
gfs_metrics_temp_speed = evaluation_temp_speed(
    forecast_dataframe=gfs_forecast,
    actual_dataframe=gfs_actual,
    model_name='GFS',
    for_temp='GFS Temp',
    for_speed='GFS Wind Speed',
    temp='Temperature',
    speed='Wind Speed',
)

# Call a function to evaluate temperature and wind speed using 4 different metrics (MAE, MSE, RMSE, R2):
ifs_metrics_temp_speed = evaluation_temp_speed(
    forecast_dataframe=ifs_forecast,
    actual_dataframe=ifs_actual,
    model_name='IFS',
    for_temp='IFS Temp',
    for_speed='IFS Wind Speed',
    temp='Temperature',
    speed='Wind Speed',
)

# Call a function to evaluate temperature and wind speed using 4 different metrics (MAE, MSE, RMSE, R2):
iconeu_metrics_temp_speed = evaluation_temp_speed(
    forecast_dataframe=iconeu_forecast,
    actual_dataframe=iconeu_actual,
    model_name='ICONEU',
    for_temp='ICONEU Temp',
    for_speed='ICONEU Wind Speed',
    temp='Temperature',
    speed='Wind Speed',
)


# Print the results for temperature and wind speed:
print('GFS WIND SPEED AND TEMPERATURE EVALUATION:\n', gfs_metrics_temp_speed)
print('IFS WIND SPEED AND TEMPERATURE EVALUATION:\n', ifs_metrics_temp_speed)
print('ICONEU WIND SPEED AND TEMPERATURE EVALUATION:\n', iconeu_metrics_temp_speed)

# ======================================================================================================================
# *** Wind Direction Evaluation ***
# ======================================================================================================================

# Call a function to evaluate wind direction using 3 different metrics (MAE, Median Absolute Error, RMSE):
wind_direction_metrics = evaluation_direction(
    gfs_data=gfs_wind_direction,
    ifs_data=ifs_wind_direction,
    iconeu_data=iconeu_wind_direction,
    gfs_angular_diff='GFS Angular Diff.',
    ifs_angular_diff='IFS Angular Diff.',
    iconeu_angular_diff='ICONEU Angular Diff.',
)

print('WIND DIRECTION EVALUATION (ALL MODELS INCLUDED):\n', wind_direction_metrics)
