import numpy as np

def calculate_inter_periph_input_and_output(
    EC, mask_inter_periph, input_is_rowsum_bool=True,
    io_based_on_nodal_degree_flag=False):
    """ECalculate the inter-periph-input and -output based on the EC matrix EC.

    INPUT:
      - EC with dimension n_rois x n_rois: effective connectivitiy matrix
      - mask_inter_periph with dimension n_rois x n_rois: defines the periph
      - input_is_rowsum_bool: boolean to determine along which dimension input and output are to
      be found
    OUTPUT:
      - input_inter_periph with dimension n_rois: all roi's input with zero for non-periphery rois
      - output_inter_periph with dimension n_rois: all roi's output with zero for non-periphery rois
    """

    if input_is_rowsum_bool:
        input_axis = 0
        output_axis = 1
    else:
        input_axis = 1
        output_axis = 0

    inter_periph_EC = np.copy(EC)
    inter_periph_EC[~mask_inter_periph] = 0
    if io_based_on_nodal_degree_flag:
        input_inter_periph = np.squeeze(np.count_nonzero(
            inter_periph_EC, axis=input_axis, keepdims=True))
        output_inter_periph = np.squeeze(np.count_nonzero(
            inter_periph_EC, axis=output_axis, keepdims=True))
    else:
        input_inter_periph = inter_periph_EC[
            :, :].sum(axis=input_axis)
        output_inter_periph = inter_periph_EC[
            :, :].sum(axis=output_axis)
    return input_inter_periph, output_inter_periph
