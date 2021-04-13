import numpy as np

def calculate_FC_with_constant_timeseries_length(filtered_ts_emp, n_shifts):
    """Calculates the covariances of filtered_ts_emp for every timeshift
    smaller than n_shifts with a constant length of timeseries n_ts_span:
    n_ts_span = n_ts_samples - n_shifts + 1, where n_ts_samples denotes the
    length of the timeseries in filtered_ts_emp.

    Input:
        filtered_ts_emp: 4D array of timeseries with dimensions:
            n_subjects x n_runs x n_rois x n_ts_samples,
        n_shifts: Upper bound (exclusive) of timeshifts for which the
            covariances are calculated. The lower bound is 0.

    Output:
        FC: Covariance matrix, which represents the FC, with dimensions:
            n_subjects, n_runs, n_shifts, n_rois, n_rois
    """
    n_subjects, n_runs, n_rois, n_ts_samples = filtered_ts_emp.shape
    FC = np.zeros([n_subjects, n_runs, n_shifts, n_rois, n_rois])
    n_ts_span = n_ts_samples - n_shifts + 1
    for i_subject in range(n_subjects):
        for i_run in range(n_runs):
            # Center the time series (around zero).
            filtered_ts_emp[i_subject, i_run, :, :] -=  \
                    np.outer(filtered_ts_emp[i_subject, i_run, :, :].
                             mean(axis=1), np.ones([n_ts_samples]))
            # Calculate covariances with various time shifts.
            for i_shift in range(n_shifts):
                for i_roi in range(n_rois):
                    FC[i_subject, i_run, i_shift, :, i_roi] = \
                    (np.dot(filtered_ts_emp[i_subject, i_run, :,
                                           0:n_ts_span],
                           filtered_ts_emp[i_subject, i_run, i_roi,
                                           i_shift:n_ts_span + i_shift])
                     / float(n_ts_span))
    return FC
