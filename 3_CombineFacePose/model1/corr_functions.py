import numpy as np
from scipy.stats import pearsonr, circmean, circvar
from scipy.signal import savgol_filter
from model1.preprocess import extract_confident_segments

all_corrs = []

def compute_corr(x, y):
	x, y = np.arctan2(np.cos(x), np.sin(x)), np.arctan2(np.cos(y), np.sin(y))
	x[x < 0] += 2 * np.pi
	y[y < 0] += 2 * np.pi
	x_m, y_m = circmean(x), circmean(y)
	upper = np.sum(np.sin(x - x_m) * np.sin(y - y_m))
	lower = np.sum(np.sin(x - x_m) ** 2) ** 0.5 * np.sum(np.sin(y - y_m) ** 2) ** 0.5
	r = upper / lower
	return r

def check_local_region(d, l, r, mono):
	if l < 0 or r > len(d):
		return None
	t = np.argmax(d[l : r])
	if abs(t - (r - l) // 2) <= 1:
		t += l
		return t
		if t > mono and t < len(d) - mono:
			if np.all(np.diff(d[t - mono + 1 : t + 1]) > 0) and np.all(np.diff(d[t : t + mono]) < 0):
				return t
	return None

# Parameters should be in number of frames. See main as an example.
def corr_peak(mother_signal, infant_signal, window_size, start_pos, max_lag, peak_picking_window = 10, savgol_window_length = 5, savgol_polyorder = 2, monotonous_size = 5):
	global all_corrs
	corrs = []
	if start_pos < max_lag or start_pos + window_size + max_lag > len(infant_signal):
		return None, None, None
	for lag in range(-max_lag, max_lag + 1):
		corr = compute_corr(mother_signal[start_pos : start_pos + window_size], infant_signal[start_pos + lag : start_pos + lag + window_size])
		corrs.append(corr)
	corr_filtered = savgol_filter(corrs, window_length = savgol_window_length, polyorder = savgol_polyorder)
	all_corrs.append(corr_filtered)
	for centre_dist in range(max_lag):
		pos_1 = max_lag - centre_dist
		pos_2 = max_lag + centre_dist
		l_1, r_2 = pos_1 - peak_picking_window // 2, pos_2 + peak_picking_window // 2
		r_1, l_2 = l_1 + peak_picking_window, r_2 - peak_picking_window
		t_1, t_2 = check_local_region(corr_filtered, l_1, r_1, monotonous_size), check_local_region(corr_filtered, l_2, r_2, monotonous_size)
		if t_1 is not None:
			return t_1 - max_lag, corr_filtered[t_1], corr_filtered
		if t_2 is not None:
			return t_2 - max_lag, corr_filtered[t_2], corr_filtered
	return None, None, None

def analysis_sequence(mother_signal_confidence, infant_signal_confidence, window_size = 5, 
                      step_size = 0.1, min_seg_length = 30, max_lag = 0.5, 
                      fps = 25, interp_peak_ratio = 0.5):
    window_size = int(round(window_size * fps))
    step_size = int(round(step_size * fps))
    min_seg_length = int(round(min_seg_length * fps))
    max_lag = int(round(max_lag * fps))
    cvt = lambda x : ("L" if x == 0 else "R")
    res = []
    for part in ["head", "arm"]:
        for m_i, (m_s, m_c) in enumerate(mother_signal_confidence[part]):
            for i_i, (i_s, i_c) in enumerate(infant_signal_confidence[part]):
                segs = extract_confident_segments((m_c + i_c) / 2, min_seg_length = min_seg_length)
                for seg_l, seg_r in segs:
                    peak_x, peaks, p_corrs = [], [], []
                    for i in range((seg_r - seg_l - window_size) // step_size):
                        peak, peak_corr, corr = corr_peak(m_s[seg_l : seg_r], i_s[seg_l : seg_r], window_size, i * step_size, max_lag)
                        if peak is not None:
                            peak_x.append(i * step_size + seg_l)
                            peaks.append(peak / fps)
                            p_corrs.append(peak_corr)
                    peak_x, peaks, p_corrs = np.array(peak_x), np.array(peaks), np.array(p_corrs)
                    if len(peaks) > interp_peak_ratio * ((seg_r - seg_l - window_size) // step_size):
                        this_res = {
							"peak_indices": (np.array(peak_x) / fps).tolist(), "peak_lags": peaks.tolist(), "peak_corrs": p_corrs.tolist(),
							"feature": part, "mother_lr": cvt(m_i), "infant_lr": cvt(i_i),
							"start_time": round(seg_l / fps, 2), "end_time": round(seg_r / fps, 2),
							"mother_init_interaction": np.sum(peaks > 0).tolist(), "infant_init_interaction": np.sum(peaks < 0).tolist(),
							"change_of_leaders": np.sum((peaks[1 : ] * peaks[ : -1]) < 0).tolist(), "correlation_intensity": np.var(peaks).tolist(),
						}
                        res.append(this_res)
    return res
