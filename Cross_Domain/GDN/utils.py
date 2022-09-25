import os
import warnings 
warnings.filterwarnings("ignore")
from shared.component_logger import component_logger as logger
from matplotlib import pyplot as plt

PLOT_DIR = "checkpoints/"

class Parameters:
    def __init__(self):
        pass

def visualize_sequence(df, cols=[], anomaly_index=[], plot_file_name='./dummy_name.PNG'):
    if not set(cols).issubset(set(df.columns)):
        return 'Error: required fields not in the dataframe'
    try:
        if not os.path.isdir(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        plot_path = PLOT_DIR + plot_file_name
        if len(cols) == 0: 
            cols = df.columns
            logger.log('Cols is not set, defaulted to df.columns')
        subplot_ncols, subplot_nrows = 1, len(cols)
        fig, axes = plt.subplots(nrows=subplot_nrows,
                                    figsize=(20*subplot_ncols,6*subplot_nrows))
        if subplot_nrows == 1:
            data = df[cols].values
            anomaly_vals = data[anomaly_index]
            axes.plot(data, 'g-', label='Normal')               
            axes.scatter(anomaly_index, anomaly_vals, color='r',
                        label='Anomaly')
            axes.grid()
            axes.legend()
        else:
            sub_axes = axes.ravel()
            for idx, col in enumerate(cols):
                data = df[col].values
                anomaly_vals = data[anomaly_index]
                sub_axes[idx].plot(data, 'g-', label='Normal')               
                sub_axes[idx].scatter(anomaly_index, anomaly_vals, color='r',
                            label='Anomaly')
                sub_axes[idx].grid()
                sub_axes[idx].legend()
                sub_axes[idx].set_ylabel(col)
        plt.savefig(plot_path)  
        plt.show()
    except Exception as e:
        logger.log("Failed to visualize: {}".format(e))
