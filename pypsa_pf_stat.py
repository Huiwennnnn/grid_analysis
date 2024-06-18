import numpy as np
def snapshot_percentage(row, over, threshold):
    if over:
        count = sum(value > threshold for value in row)
    else:
        count = sum(value < threshold for value in row)
    row_length = len(row)
    return count / row_length * 100


def pf_undervoltage(net):
    return net.buses_t.v_mag_pu.apply(lambda row: snapshot_percentage(row, False, 0.97), axis=1)


def pf_overvoltage(net):
    return net.buses_t.v_mag_pu.apply(lambda row: snapshot_percentage(row, True, 1.03), axis=1)


def opt_line_loading(net):
    return abs(net.lines_t.p0) / net.lines.s_nom * 100


def opt_line_overloading_stat(net):
    line_loading = opt_line_loading(net)
    overload_percentage = line_loading.apply(lambda row: snapshot_percentage(row, True, 100), axis=1)
    return overload_percentage


def pf_line_loading(net):
    return np.sqrt(net.lines_t.p0 ** 2 + net.lines_t.q0 ** 2) / net.lines.s_nom * 100


def pf_line_overloading_stat(net):
    line_loading = pf_line_loading(net)
    overload_percentage = line_loading.apply(lambda row: snapshot_percentage(row, True, 100), axis=1)
    return overload_percentage


def opt_trafo_loading(net):
    return abs(net.transformers_t.p0) / net.transformers.s_nom * 100


def opt_trafo_overloading_stat(net):
    trafo_loading = opt_trafo_loading(net)
    overload_percentage = trafo_loading.apply(lambda row: snapshot_percentage(row, True, 100), axis=1)
    return overload_percentage


def pf_trafo_loading(net):
    return np.sqrt(net.transformers_t.p0 ** 2 + net.transformers_t.q0 ** 2) / net.transformers.s_nom*100


def pf_trafo_overloading_stat(net):
    trafo_loading = pf_trafo_loading(net)
    overload_percentage = trafo_loading.apply(lambda row: snapshot_percentage(row, True, 100), axis=1)
    return overload_percentage

def replace_small_values(val, threshold=1e-9):
    if abs(val) < threshold:
        return 0
    else:
        return val