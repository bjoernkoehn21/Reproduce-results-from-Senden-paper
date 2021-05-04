import numpy as np

def calculate_inter_rc_input_and_output(
    EC, mask_inter_rc, input_is_rowsum_bool=True,
    io_based_on_nodal_degree_flag=False):
    """Calculate the inter-rc-input and -output based on the EC matrix EC.

    INPUT:
      - EC with dimension n_rois x n_rois: effective connectivitiy matrix
      - mask_inter_rc with dimension n_rois x n_rois: defines the rc
      - input_is_rowsum_bool: boolean to determine along which dimension input and output are to
      be found
    OUTPUT:
      - input_inter_rc with dimension n_rois: all roi's input with zero for non-rc rois
      - output_inter_rc with dimension n_rois: all roi's output with zero for non-rc rois
    """

    if input_is_rowsum_bool:
        input_axis = 0
        output_axis = 1
    else:
        input_axis = 1
        output_axis = 0

    inter_rc_EC = np.copy(EC)
    inter_rc_EC[~mask_inter_rc] = 0
    if io_based_on_nodal_degree_flag:
        input_inter_rc= np.squeeze(np.count_nonzero(
            inter_rc_EC, axis=input_axis, keepdims=True))
        output_inter_rc= np.squeeze(np.count_nonzero(
            inter_rc_EC, axis=output_axis, keepdims=True))
    else:
        input_inter_rc= inter_rc_EC[
            :, :].sum(axis=input_axis)
        output_inter_rc= inter_rc_EC[
            :, :].sum(axis=output_axis)
    return input_inter_rc, output_inter_rc
