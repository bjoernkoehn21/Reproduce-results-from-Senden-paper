import numpy as np

def calculate_FC_shift(filtered_ts_emp, time_shift):
    """Calculates the covariances of filtered_ts_emp for the given timeshift
    time_shift. The covariances are calculated with the maximum length of
    timeseries possible for the given time shift:
    n_ts_span = n_ts_samples - time_shift, where n_ts_samples denotes the
    length of the timeseries in filtered_ts_emp.

    Input:
        filtered_ts_emp: 4D array of timeseries with dimensions:
            n_subjects x n_runs x n_rois x n_ts_samples,
        time_shift: Timeshift for which the covariances are calculated.

    Output:
        FC: Covariance matrix, which represents the FC, with dimensions:
            n_subjects, n_runs, n_rois, n_rois
    """
    n_subjects, n_runs, n_rois, n_ts_samples = filtered_ts_emp.shape
    FC_shift = np.zeros([n_subjects, n_runs, n_rois, n_rois])
    FC_Reuters = np.zeros([n_subjects, n_runs, n_rois, n_rois])
    n_ts_span = n_ts_samples - time_shift
    for i_subject in range(n_subjects):
        for i_run in range(n_runs):
            # Center the time series (around zero).
            filtered_ts_emp[i_subject, i_run, :, :] -=  \
                    np.outer(filtered_ts_emp[i_subject, i_run, :, :].
                             mean(axis=1), np.ones([n_ts_samples]))
            # Calculate covariances with time shift time_shift.
            for i_roi in range(n_rois):
                FC_shift[i_subject, i_run, :, i_roi] = \
                np.dot(filtered_ts_emp[i_subject, i_run, :,
                                       :n_ts_span],
                       filtered_ts_emp[i_subject, i_run, i_roi,
                                       time_shift:]) / float(n_ts_span)
    return FC_shift
