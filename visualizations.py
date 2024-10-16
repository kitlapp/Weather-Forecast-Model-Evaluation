import matplotlib.pyplot as plt  # For basic plotting.
import seaborn as sns  # For enhanced plotting.


def create_vertical_barplot(dataframe, x_axis, y_axis, y_lim, title, x_label, x_ticks_rot, y_label, y_ticks_rot):
    """
        Creates a vertical bar plot from the specified DataFrame.

        Params:
            dataframe (pd.DataFrame): The DataFrame containing the data for the bar plot.
            x_axis (str): The column name for the x-axis values.
            y_axis (str): The column name for the y-axis values.
            y_lim (float): The upper limit for the y-axis.
            title (str): The title of the bar plot.
            x_label (str): The label for the x-axis.
            x_ticks_rot (int): The rotation angle for the x-axis tick labels.
            y_label (str): The label for the y-axis.
            y_ticks_rot (int): The rotation angle for the y-axis tick labels.

        Returns:
            matplotlib.axes.Axes: The axes object of the created bar plot.
        """
    plt.figure(figsize=(10, 5), dpi=130)
    ax = sns.barplot(data=dataframe, x=x_axis, y=y_axis)
    ax.set_ylim(0, y_lim)
    ax.grid(True, which='major', axis='y')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    plt.xticks(rotation=x_ticks_rot)
    ax.set_ylabel(y_label)
    plt.yticks(rotation=y_ticks_rot)
    plt.show()
    return ax
