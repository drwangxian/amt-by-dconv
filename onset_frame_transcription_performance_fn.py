from scipy.ndimage.filters import maximum_filter1d
import librosa
from functools import partial


def onset_frame_transcription_performance_fn(logits_dict, ref_intervals, ref_notes, sr, hop_size):
    """Given logits of onset and frame, compute transcription metrics. Specifically,
    we compute three types of metrics, namely, note level metrics considering only onsets,
    note level metrics considering both onsets and offsets, and the frame level metrics based on
    the predicted notes.

    Parameters
    ----------
    logits_dict: dict with keys onset and frame, each value is of type np.ndarray, dtype np.float32, and shape (n, 2)
        A dictionary with keys onset and frame to store onset and frame logits
    ref_intervals: np.ndarray, dtype=np.float32, shape=(m, 2)
        Array of reference note time intervals (onset and offset times in seconds)
    ref_notes: np.ndarray, dtype=np.uint8, shape=(m,)
        Array of reference notes which are standard midi notes ranging from 21 to 108
    sr: int
        sampling rate
    hop_size: int
        hop size in terms of samples

    Returns
    -------
    A dictionary with keys with, without and frame where
    without: note level metrics (including precision, recall, f1, and overlap) considering only onsets
    with: note level metrics (including precision, recall, f1, and overlap) considering both onsets and offsets
    frame: frame level metrics (including precision, recall, and f1) based on the predicted notes
    """

    def _find_peak_locations_fn(logits, size, threshold):

        assert size % 2 == 1
        assert logits.ndim == 2 and logits.shape[1] == 88
        logits_max = maximum_filter1d(logits, size=size, axis=0, mode='constant')
        assert logits_max.shape == logits.shape
        logits_peak_ids = np.logical_and(logits == logits_max, logits > threshold)

        # Due to numerical precision, there could be multiple peaks within a window.
        # In this case, we use the first peak and remove the other peaks.
        num_frames = len(logits)
        hs = (size - 1) // 2
        for pitch in xrange(88):
            for frame_idx in xrange(num_frames - hs):
                if logits_peak_ids[frame_idx, pitch]:
                    for fidx in xrange(hs):
                        if logits_peak_ids[frame_idx + 1 + fidx, pitch]:
                            logits_peak_ids[frame_idx + 1 + fidx, pitch] = False

        logits_peak_ids = np.where(logits_peak_ids)

        return logits_peak_ids

    def _frame_level_performance_fn(note_seq_pred, ref_intervals, ref_pitches, sr, hop_size):

        def _times_to_frames_fn(sr, hop_size, start_time, end_time):
            sr = int(sr)
            hop_size = int(hop_size)
            assert hop_size % 2 == 0
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            start_frame = (start_sample + hop_size // 2) // hop_size
            end_frame = (end_sample + hop_size // 2 - 1) // hop_size
            return start_frame, end_frame + 1

        t2f_fn = partial(_times_to_frames_fn, sr=sr, hop_size=hop_size)

        pred_frames = max(note[1] for note_seq in note_seq_pred for note in note_seq)
        ref_duration = np.max(ref_intervals)
        _, ref_frames = t2f_fn(start_time=0., end_time=ref_duration)
        num_frames = max(pred_frames, ref_frames)
        pred_labels = np.zeros((num_frames, 88), dtype=np.bool_)
        ref_labels = np.zeros_like(pred_labels)

        for pitch, note_seq in enumerate(note_seq_pred):
            for note in note_seq:
                pred_labels[note[0]:note[1], pitch] = True

        for pitch, (s, e) in zip(ref_pitches, ref_intervals):
            assert 21 <= pitch <= 108
            sf, ef = t2f_fn(start_time=s, end_time=e)
            assert ef > sf
            ref_labels[sf:ef, pitch - 21] = True

        tps = np.sum(np.logical_and(pred_labels, ref_labels))
        p = 1. * tps / (np.sum(pred_labels) + 1e-7)
        r = 1. * tps / (np.sum(ref_labels) + 1e-7)
        f = 2. * p * r / (p + r + 1e-7)

        return p, r, f

    assert len(ref_intervals) == len(ref_notes)
    assert ref_intervals.ndim == 2 and ref_intervals.shape[1] == 2
    assert ref_notes.ndim == 1
    assert ref_intervals.dtype == np.float32 and ref_notes.dtype == np.uint8

    logits_onset = logits_dict['onset']
    logits_frame = logits_dict['frame']
    assert len(logits_onset) == len(logits_frame)
    num_frames = len(logits_frame)

    onsets_peak_picked = np.empty_like(logits_onset, dtype=np.bool_)
    onsets_peak_picked.fill(False)
    onsets_peak_picked[_find_peak_locations_fn(logits_onset, size=5, threshold=0.)] = True
    onsets_peak_picked[num_frames - 2:] = False
    onsets_peak_picked[:2] = False

    # exclude false positives onsets
    logits_frame = logits_dict['frame'] > 0.
    for pitch in xrange(88):
        for frame_idx in xrange(num_frames):
            if onsets_peak_picked[frame_idx, pitch]:
                onsets_peak_picked[frame_idx, pitch] = logits_frame[frame_idx, pitch] \
                                                       or logits_frame[frame_idx + 1, pitch]

    for pitch in xrange(88):
        for frame_idx in xrange(num_frames):
            if onsets_peak_picked[frame_idx, pitch]:
                logits_frame[frame_idx - 1, pitch] = False
    logits_frame[num_frames - 1] = False

    note_seq_pred = [[] for _ in xrange(88)]
    for pitch in xrange(88):
        onset_poses = np.where(onsets_peak_picked[:, pitch])[0]
        num_onsets = len(onset_poses)
        for onset_idx, onset_frame in enumerate(onset_poses):
            if onset_idx == num_onsets - 1:
                next_onset_frame = num_frames
            else:
                next_onset_frame = onset_poses[onset_idx + 1]

            assert next_onset_frame - onset_frame >= 3
            assert logits_frame[onset_frame, pitch] or logits_frame[onset_frame + 1, pitch]
            assert not logits_frame[onset_frame - 1, pitch]
            assert not logits_frame[next_onset_frame - 1, pitch]

            for offset_frame in xrange(onset_frame + 1, next_onset_frame):
                if not logits_frame[offset_frame, pitch]:
                    break
            else:
                assert False

            if logits_frame[onset_frame, pitch]:
                onset_frame -= 1
            if offset_frame == next_onset_frame - 1:
                offset_frame += 1
            offset_frame -= 1
            assert offset_frame > onset_frame
            if offset_frame == onset_frame + 1:
                offset_frame += 1
            assert offset_frame < next_onset_frame

            note_seq_pred[pitch].append([onset_frame, offset_frame])

    pred_intervals = []
    pred_pitches = []
    for pitch, note_seq in enumerate(note_seq_pred):
        for note in note_seq:
            pred_pitches.append(pitch + 21)
            start_time = 1. * note[0] * hop_size / sr
            end_time = 1. * note[1] * hop_size / sr
            pred_intervals.append([start_time, end_time])

    if pred_intervals:
        pred_intervals = np.asarray(pred_intervals, dtype=np.float32).reshape(-1, 2)
        pred_pitches = np.asarray(pred_pitches, dtype=np.uint8).reshape(-1)

        prfo_offset = precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=librosa.midi_to_hz(ref_notes),
            est_intervals=pred_intervals,
            est_pitches=librosa.midi_to_hz(pred_pitches),
            offset_ratio=.2
        )

        prfo = precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=librosa.midi_to_hz(ref_notes),
            est_intervals=pred_intervals,
            est_pitches=librosa.midi_to_hz(pred_pitches),
            offset_ratio=None
        )

        prf_frame = _frame_level_performance_fn(
            note_seq_pred=note_seq_pred,
            ref_intervals=ref_intervals,
            ref_pitches=ref_notes,
            sr=sr,
            hop_size=hop_size
        )
        tmp = dict()
        tmp['with'] = prfo_offset
        tmp['without'] = prfo
        tmp['frame'] = prf_frame
    else:
        warnings.warn("Estimated notes are empty.")
        tmp = dict()
        tmp['with'] = 0., 0., 0., 0.
        tmp['without'] = 0., 0., 0., 0.
        tmp['frame'] = 0., 0., 0.

    return tmp