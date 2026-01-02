import re
import numpy as np
def standardize_run_label(run_str):
    """
    Converts 'run-1' to 'run-01', 'run-10' stays as 'run-10', etc.
    """
    match = re.match(r"run-(\d+)", run_str)
    if match:
        run_num = int(match.group(1))
        return f"run-{run_num:02d}"
    else:
        raise ValueError(f"Invalid run format: {run_str}")


def glm_confounds_construction(df_confounds):
    # note this is a hand selection of good confounds (IMO) for task GLMs
    timeseries_confounds = []
    # csf signals
    timeseries_confounds.append(df_confounds['csf'])
    timeseries_confounds.append(df_confounds['csf_power2'])
    timeseries_confounds.append(df_confounds['csf_derivative1']) # nan
    timeseries_confounds.append(df_confounds['csf_derivative1_power2']) # nan
    # wm signals
    timeseries_confounds.append(df_confounds['white_matter'])
    timeseries_confounds.append(df_confounds['white_matter_power2'])
    timeseries_confounds.append(df_confounds['white_matter_derivative1']) # nan
    timeseries_confounds.append(df_confounds['white_matter_derivative1_power2']) # nan
    # motion signals
    timeseries_confounds.append(df_confounds['trans_x'])
    timeseries_confounds.append(df_confounds['trans_x_power2'])
    timeseries_confounds.append(df_confounds['trans_x_derivative1'])
    timeseries_confounds.append(df_confounds['trans_x_derivative1_power2'])
    timeseries_confounds.append(df_confounds['trans_y'])
    timeseries_confounds.append(df_confounds['trans_y_power2'])
    timeseries_confounds.append(df_confounds['trans_y_derivative1'])
    timeseries_confounds.append(df_confounds['trans_y_derivative1_power2'])
    timeseries_confounds.append(df_confounds['trans_z'])
    timeseries_confounds.append(df_confounds['trans_z_power2'])
    timeseries_confounds.append(df_confounds['trans_z_derivative1'])
    timeseries_confounds.append(df_confounds['trans_z_derivative1_power2'])
    timeseries_confounds.append(df_confounds['rot_x'])
    timeseries_confounds.append(df_confounds['rot_x_power2'])
    timeseries_confounds.append(df_confounds['rot_x_derivative1'])
    timeseries_confounds.append(df_confounds['rot_x_derivative1_power2'])
    timeseries_confounds.append(df_confounds['rot_y'])
    timeseries_confounds.append(df_confounds['rot_y_power2'])
    timeseries_confounds.append(df_confounds['rot_y_derivative1'])
    timeseries_confounds.append(df_confounds['rot_y_derivative1_power2'])
    timeseries_confounds.append(df_confounds['rot_z'])
    timeseries_confounds.append(df_confounds['rot_z_power2'])
    timeseries_confounds.append(df_confounds['rot_z_derivative1'])
    timeseries_confounds.append(df_confounds['rot_z_derivative1_power2'])
    #
    timeseries_confounds = np.asarray(timeseries_confounds).T # change to time x confounds
    return timeseries_confounds