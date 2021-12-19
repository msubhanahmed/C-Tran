import numpy as np

from resampling import lp_ros, lp_rus, ml_ros, ml_rus


def __resample(x, y, resample_func, oversample, percentage=10):
    idxs = resample_func(np.array(y), percentage)

    if oversample:
        return x.append(x.iloc[idxs, :], ignore_index=True), y.append(y.iloc[idxs, :], ignore_index=True)
    else:
        mask = np.ones(y.shape[0], dtype=bool)
        mask[idxs]=False
        return x.iloc[mask, :], y.iloc[mask, :]

def resample_dataset(x, y, algorithm, percentage=10):
    if algorithm == 'lp_rus':
        return __resample(x, y, lp_rus.LP_RUS, False, percentage)
    if algorithm == 'lp_ros':
        return __resample(x, y, lp_ros.LP_ROS, True, percentage)
    if algorithm == 'lp_rus+lp_ros':
        x_new, y_new = __resample(x, y, lp_rus.LP_RUS, False, percentage)
        return __resample(x_new, y_new, lp_ros.LP_ROS, True, percentage)
    if algorithm == 'ml_rus':
        return __resample(x, y, ml_rus.ML_RUS, False, percentage)
    if algorithm == 'ml_ros':
        return __resample(x, y, ml_ros.ML_ROS, True, percentage)
    if algorithm == 'ml_rus+ml_ros':
        x_new, y_new = __resample(x, y, ml_rus.ML_RUS, False, percentage)
        return __resample(x_new, y_new, ml_ros.ML_ROS, True, percentage)

    print('Resample algorithm \'{}\' not found. Please check spelling'.format(algorithm))
    return x, y
