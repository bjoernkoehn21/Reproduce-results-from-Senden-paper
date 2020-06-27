import os
import itertools
import scipy.io
import scipy.stats as stt
import numpy as np
import matplotlib.pyplot as plt
from mou_model import MOU

_RES_DIR = 'model_parameter/'
_I_REST_RUN = 0
_I_NBACK_RUN = 1
_I_NO_TIMESHIFT = 0
_I_ONE_TIMESHIFT = 1
_SUBJECT_AXIS = 0

plt.close('all')

## Create a local folder to store results.
if not os.path.exists(_RES_DIR):
    print('created directory:', _RES_DIR)
    os.makedirs(_RES_DIR)

## Read in data and define constants.
fMRI_data_and_labels = scipy.io.loadmat('data/DATA_TASK_3DMOV_HP_CSF_WD.mat')
regionalized_preprocessed_fMRI_data = fMRI_data_and_labels['TASKEC'][0][0]
roi_labels = fMRI_data_and_labels['ROIlbls'][0]
rest_run_data = regionalized_preprocessed_fMRI_data['Rest']
n_back_run_data = regionalized_preprocessed_fMRI_data['nBack']
flanker_run_data = regionalized_preprocessed_fMRI_data['Flanker']
m_rotation_run_data = regionalized_preprocessed_fMRI_data['mRotation']
odd_man_out_run_data = regionalized_preprocessed_fMRI_data['OddManOut']

n_subjects = rest_run_data.shape[2]
n_runs = len(fMRI_data_and_labels['TASKEC'][0][0])
n_rois = rest_run_data.shape[1]
n_ts_samples = rest_run_data.shape[0]
# Structure data to match the format used at
# https://github.com/mb-BCA/notebooks_review2019/blob/master/1_MOUEC_Estimation.ipynb
# to enhance comparability.
filtered_ts_emp = np.zeros([n_subjects, n_runs, n_rois, n_ts_samples])
run = list(regionalized_preprocessed_fMRI_data.dtype.fields.keys())
for k in range(len(run)):
    filtered_ts_emp[:, k, :, :] = np.transpose(
        regionalized_preprocessed_fMRI_data[run[k]], (2, 1, 0))

## Calculate functional connectivity (BOLD covariances) [Q0 und Q1].
# time_shift = np.arange(4, dtype=float) # for autocovariance plots
time_shift = np.arange(2, dtype=float)
n_shifts = len(time_shift)

FC_emp = np.zeros([n_subjects, n_runs, n_shifts, n_rois, n_rois])
n_ts_span = n_ts_samples - n_shifts + 1
for i_subject in range(n_subjects):
    for i_run in range(n_runs):
        # Center the time series (around zero).
        filtered_ts_emp[i_subject, i_run, :, :] -=  \
                np.outer(filtered_ts_emp[i_subject, i_run, :, :].mean(1),
                         np.ones([n_ts_samples]))
        # Calculate covariances with various time shifts.
        for i_shift in range(n_shifts):
            FC_emp[i_subject, i_run, i_shift, :, :] = \
                np.tensordot(filtered_ts_emp[i_subject, i_run, :,
                                             0:n_ts_span],
                             filtered_ts_emp[i_subject, i_run, :,
                                             i_shift:n_ts_span+i_shift],
                             axes=(1, 1)) / float(n_ts_span - 1)

rescale_FC_factor = (0.5 / FC_emp[:, _I_REST_RUN, _I_NO_TIMESHIFT, :,
                                  :].diagonal(axis1=1, axis2=2).mean())
FC_emp *= rescale_FC_factor
# filtered_ts_emp /= np.sqrt(0.135) # Rescale to get the same order of magnitude for locale variability as in paper.

print('most of the FC values should be between 0 and 1')
print('mean FC0 value:', FC_emp[:, :, _I_NO_TIMESHIFT, :, :].mean(),
      FC_emp.mean())
print('max FC0 value:', FC_emp[:, :, _I_NO_TIMESHIFT, :, :].max())
print('mean BOLD variance (diagonal of each FC0 matrix):',
      FC_emp[:, :, _I_NO_TIMESHIFT, :, :].diagonal(axis1=2, axis2=3).mean())
print('rescaleFactor: ' + str(rescale_FC_factor))
# Show distibution of FC0 values.
plt.figure()
plt.hist(FC_emp[:, :, _I_NO_TIMESHIFT, :, :].flatten(),
         bins=np.linspace(-1, 5, 30))
plt.xlabel('FC0 value', fontsize=14)
plt.ylabel('matrix element count', fontsize=14)
plt.title('distribution of FC0 values')
# Show FC0 averaged over subjects first run (rest).
plt.figure()
FC_avg_over_subj = FC_emp[:, _I_REST_RUN,
                          _I_NO_TIMESHIFT, :, :].mean(axis=_SUBJECT_AXIS)
plt.imshow(FC_avg_over_subj, origin='lower', cmap='Blues', vmax=1, vmin=0)
plt.colorbar()
plt.xlabel('target ROI', fontsize=14)
plt.ylabel('source ROI', fontsize=14)
plt.title('FC0 (functional connectivity with no time lag)')
# Show FC1 averaged over subjects first run (rest).
plt.figure()
FC_avg_over_subj = FC_emp[:, _I_REST_RUN,
                          _I_ONE_TIMESHIFT, :, :].mean(axis=_SUBJECT_AXIS)
plt.imshow(FC_avg_over_subj, origin='lower', cmap='Blues', vmax=1, vmin=0)
plt.colorbar()
plt.xlabel('target ROI', fontsize=14)
plt.ylabel('source ROI', fontsize=14)
plt.title('FC1 (functional connectivity with time lag 1TR)')
# Show the autocovariance for the first run (rest).
ac = FC_emp.diagonal(axis1=3, axis2=4)
plt.figure()
ac_avg_over_subj = np.log(np.maximum(ac[:, _I_REST_RUN, :, :].
                                     mean(axis=_SUBJECT_AXIS), np.exp(-4.0)))
plt.plot(range(n_shifts), ac_avg_over_subj)
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('rest', fontsize=16)
plt.ylim((-3, 0))
plt.xlim((0, 3))
# Show the autocovariance for the 2nd run (nBack).
plt.figure()
ac_avg_over_subj = np.log(np.maximum(ac[:, _I_NBACK_RUN, :, :].
                                     mean(axis=_SUBJECT_AXIS), np.exp(-3.1)))
plt.plot(range(n_shifts), ac_avg_over_subj)
plt.xlabel('time lag', fontsize=14)
plt.ylabel('log autocovariance', fontsize=14)
plt.title('nBack', fontsize=16)
plt.ylim((-3, 0))
plt.xlim((0, 3))

## Include structural connectivity.
# Load the binary structural connectivity matrix.
mask_EC = np.array(scipy.io.loadmat('data/BINARY_EC_MASK.mat')
                   ['grouped_umcu50_60percent'], dtype=bool)
# Enforce hermispheric connections.
for i in range(int(n_rois/2)):
    mask_EC[i, int(n_rois/2)+i] = True
    mask_EC[int(n_rois/2)+i, i] = True
# Visualise the structural connectivity mask.
plt.figure()
plt.imshow(mask_EC, origin='lower')
plt.xlabel('target ROI', fontsize=14)
plt.ylabel('source ROI', fontsize=14)
plt.title('Mask for existing connections', fontsize=12)

## Calculate EC-matrix.
# Construct diagonal mask for input noise matrix
# (here, no input cross-correlation).
mask_Sigma = np.eye(n_rois, dtype=bool)
# Run the model optimization.
# Initialize the source arrays.

# Jacobian (off-diagonal elements = EC)
J_mod = np.zeros([n_subjects, n_runs, n_rois, n_rois])
# Local variance (input covariance matrix, chosen to be diagonal)
Sigma_mod = np.zeros([n_subjects, n_runs, n_rois, n_rois])
# Model error
dist_mod = np.zeros([n_subjects, n_runs])
# Approximation of variance about the fitted data (FC covariance matrices)
R2_mod = np.zeros([n_subjects, n_runs])
# Between-region EC matrix
C_mod = np.zeros([n_subjects, n_runs, n_rois, n_rois])

mou_model = MOU()

for i_subject in range(n_subjects):
    for i_run in range(n_runs):
        # Run the estimation of model parameters, for all sessions.
        # All parameters/restrictions not explicitly passed, have the
        # correct defaults in fit_LO@MOU.
        mou_model.fit(filtered_ts_emp[i_subject, i_run, :, :].T,
                      mask_Sigma=mask_Sigma, mask_C=mask_EC)
        # Organize the optimization results into arrays.
        # Extract Jacobian of the model.
        J_mod[i_subject, i_run, :, :] = mou_model.J
        # Extract noise (auto-)covariance matrix.
        Sigma_mod[i_subject, i_run, :, :] = mou_model.Sigma
        # Extract the matrix distance between the empirical objective
        # covariances and their model counterparts
        # (normalized for each objective matrix).
        dist_mod[i_subject, i_run] = mou_model.d_fit['distance']
        # The squared Pearson correlation is taken as an approximation
        # of the variance.
        R2_mod[i_subject, i_run] = mou_model.d_fit['correlation']**2
        # The between-region EC matrix of the model
        C_mod[i_subject, i_run, :, :] = mou_model.get_C()

        print('sub / run:', i_subject, i_run, ';\t model error, R2:',
              dist_mod[i_subject, i_run], R2_mod[i_subject, i_run])
# Store the results in files.
np.save(_RES_DIR + 'FC_emp.npy',
        FC_emp)  # Empirical spatiotemporal FC
np.save(_RES_DIR + 'mask_EC.npy',
        mask_EC)  # Mask of optimized connections
np.save(_RES_DIR + 'mask_Sigma.npy',
        mask_Sigma)  # Mask of optimized Sigma elements
np.save(_RES_DIR + 'Sigma_mod.npy',
        Sigma_mod)  # Estimated Sigma matrices
np.save(_RES_DIR + 'dist_mod.npy',
        dist_mod)  # Model error
np.save(_RES_DIR + 'J_mod.npy',
        J_mod)  # Estimated Jacobian, EC + inverse time const. on diag.
print('\nFinished.')

# Plot C-matrix for resting state data.
plt.figure()
plt.imshow(C_mod[:, _I_REST_RUN, :, :].mean(axis=_SUBJECT_AXIS),
           origin='lower', cmap='Reds')
plt.colorbar()
plt.xlabel('target ROI', fontsize=14)
plt.ylabel('source ROI', fontsize=14)
plt.title('effective connectivity C_{ij}')
plt.show()

## Calculate local variability for rich club and periphery.
mean_rc_var = np.zeros([n_runs])
mean_periph_var = np.zeros([n_runs])
conf_int_rc = np.zeros([n_runs, 2])
conf_int_periph = np.zeros([n_runs, 2])
# Create a 1D-mask for rich club regions.
mask_rc = np.zeros(n_rois, dtype=bool)
indexes_rich_club = [23, 26, 27, 57, 60, 61]
mask_rc[indexes_rich_club] = True
print('rich club regions: '
      + str(np.concatenate(roi_labels[indexes_rich_club]).tolist()))

for i_run in range(n_runs):
    local_var = Sigma_mod[:, i_run, :, :].diagonal(axis1=1, axis2=2)
    rc_var = local_var[:, mask_rc].mean(axis=1)
    periph_var = local_var[:,  ~mask_rc].mean(axis=1)
    mean_rc_var[i_run] = rc_var.mean()
    mean_periph_var[i_run] = periph_var.mean()
    sigma_rc_var = rc_var.std(ddof=1)
    sigma_periph_var = periph_var.std(ddof=1)
    conf_int_rc[i_run, :] = stt.norm.interval(0.95,
                                              loc=mean_rc_var[i_run],
                                              scale=sigma_rc_var)
    conf_int_periph[i_run, :] = stt.norm.interval(0.95,
                                                  loc=mean_periph_var[i_run],
                                                  scale=sigma_periph_var)
print('Mittel der lokalen Variabilität (rich club): ' + str(mean_rc_var))
print('Mittel der lokalen Variabilität (periphery): ' + str(mean_periph_var))
print('95% Konfidenz Interval (rich cluc): ' + str(conf_int_rc))
print('95% Konfidenz Interval (periphery): ' + str(conf_int_periph))

## Calculate the input-output ratio.
# Create a 2D-mask for rich club regions.
mask_inter_rc = np.zeros([n_rois, n_rois], dtype=bool)
# The entries on the diagonal of C are 0 anyway, so that they can be
# ignored when it comes to the mask:
# mask_inter_rc[indexes_rich_club, indexes_rich_club] = True
rc_ind_combin = np.array(list(itertools.permutations(indexes_rich_club, 2))).T
mask_inter_rc[rc_ind_combin[0], rc_ind_combin[1]] = True

mean_rc_io = np.zeros([n_runs])
mean_periph_io = np.zeros([n_runs])
for i_run in range(n_runs):
    # Examine input-output ratio ignoring inter-rich-club connections.
    no_rc_connections_C = C_mod[:, i_run, :, :]
    no_rc_connections_C[:, mask_inter_rc] = 0
    roi_input = no_rc_connections_C[:, :, :].sum(axis=1)
    roi_output = no_rc_connections_C[:, :, :].sum(axis=2)
    io_rc = (roi_input[:, mask_rc].sum(axis=1) /
                              roi_output[:, mask_rc].sum(axis=1))
    io_periph = (roi_input[:, ~mask_rc].sum(axis=1) /
                               roi_output[:, ~mask_rc].sum(axis=1))
    mean_rc_io[i_run] = io_rc.mean()
    mean_periph_io[i_run] = io_periph.mean()


    sigma_io_rc = io_rc.std(ddof=1)
    sigma_io_periph = io_periph.std(ddof=1)
    conf_int_rc[i_run, :] = stt.norm.interval(0.95,
                                              loc=mean_rc_io[i_run],
                                              scale=sigma_io_rc)
    conf_int_periph[i_run, :] = stt.norm.interval(0.95,
                                                  loc=mean_periph_io[i_run],
                                                  scale=sigma_io_periph)
print('input-output ratio rich club: ', str(mean_rc_io))
print('input-output ratio periphery: ', str(mean_periph_io))
print('95% Konfidenz Interval (rich cluc): ' + str(conf_int_rc))
print('95% Konfidenz Interval (periphery): ' + str(conf_int_periph))
