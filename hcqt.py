"""
This script implements the fully convolutional acoustic model listed in Table II.

How to use this code
1. Download the MAPS dataset and generate the HCQT spectrograms.
2. Create a folder, e.g., maps/hcqt, and copy this script to the folder. By default, the checkpoints will be saved in
   folder ./saved_model, and the statistics and other information will be saved in folder ./tb_d0. You can view the
   outputs with tensorboard.
3. Configure the following parameters:
    DEBUG: in {True, False}. If True, will run in a debug mode where only very few recordings will be run. The debug
           mode enables you to quickly check if the script can run correctly.
    GPU_ID: in {0, 1, ..., n - 1} where n is the number of GPUs available.
4. Refer to class Config for more options, e.g., continue training from a saved checkpoint, or run in inference mode.
5. This script is not fully commented. It is better to first familiarize yourself with script comp_acoustic_models.py.
"""


from __future__ import print_function
import numpy as np

DEBUG = False
GPU_ID = 0

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import re

from argparse import Namespace
import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
import datetime
import magenta.music
import soundfile
import mido


class MiscFns(object):
    """Miscellaneous functions"""

    @staticmethod
    def filename_to_id(filename):
        """Translate a .wav or .mid path to a MAPS sequence id."""
        return re.match(r'.*MUS-(.+)_[^_]+\.\w{3}',
                        os.path.basename(filename)).group(1)

    @staticmethod
    def times_to_frames_fn(start_time, end_time):
        sr = 44100
        spec_stride = 512
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        start_frame = (start_sample + spec_stride // 2) // spec_stride
        end_frame = (end_sample + spec_stride // 2 - 1) // spec_stride

        return start_frame, end_frame + 1

    @staticmethod
    def label_fn(mid_file_name, num_frames):
        frame_matrix = np.zeros((num_frames, 88), dtype=np.bool_)
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)
        for note in note_seq.notes:
            assert 21 <= note.pitch <= 108
            note_start_frame, note_end_frame = MiscFns.times_to_frames_fn(
                start_time=note.start_time,
                end_time=note.end_time
            )
            frame_matrix[note_start_frame: note_end_frame, note.pitch - 21] = True

        return frame_matrix

    @staticmethod
    def split_train_valid_and_test_files_fn():
        test_dirs = ['ENSTDkCl_2/MUS', 'ENSTDkAm_2/MUS']
        train_dirs = ['AkPnBcht_2/MUS', 'AkPnBsdf_2/MUS', 'AkPnCGdD_2/MUS', 'AkPnStgb_2/MUS',
                      'SptkBGAm_2/MUS', 'SptkBGCl_2/MUS', 'StbgTGd2_2/MUS']
        maps_dir = os.environ['maps']

        test_files = []
        for directory in test_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            test_files += wav_files

        test_ids = set([MiscFns.filename_to_id(wav_file) for wav_file in test_files])
        assert len(test_ids) == 53

        training_files = []
        validation_files = []
        for directory in train_dirs:
            path = os.path.join(maps_dir, directory)
            path = os.path.join(path, '*.wav')
            wav_files = glob.glob(path)
            for wav_file in wav_files:
                me_id = MiscFns.filename_to_id(wav_file)
                if me_id not in test_ids:
                    training_files.append(wav_file)
                else:
                    validation_files.append(wav_file)

        assert len(training_files) == 139 and len(test_files) == 60 and len(validation_files) == 71

        return dict(training=training_files, test=test_files, validation=validation_files)

    @staticmethod
    def gen_split_list_fn(num_frames, snippet_len):
        split_frames = range(0, num_frames + 1, snippet_len)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])
        start_end_frame_pairs = [list(it) for it in start_end_frame_pairs]

        return start_end_frame_pairs

    @staticmethod
    def load_np_array_from_file_fn(file_name):
        with open(file_name, 'rb') as fh:
            first_line = str(fh.readline()).split()
            rec_name = first_line[0]
            dtype = first_line[1]
            dim = first_line[2:]
            dim = [int(_item) for _item in dim]
            output = np.frombuffer(fh.read(), dtype=dtype).reshape(*dim)

            return rec_name, output

    @staticmethod
    def num_samples_to_num_frames_fn(num_samples):
        assert isinstance(num_samples, (int, long))
        h = 512
        num_frames = (num_samples + h - 1) // h

        return num_frames

    @staticmethod
    def array_to_table_tf_fn(tf_array, header, scope, title, names, precision=None):
        tf_array = tf.convert_to_tensor(tf_array)
        assert tf_array._rank() == 2
        num_examples = tf_array.shape[0].value
        num_fields = tf_array.shape[1].value
        assert num_examples is not None
        assert num_fields is not None
        assert isinstance(header, list)
        assert len(header) == num_fields
        header = ['id', 'name'] + header
        header = tf.constant(header)
        assert isinstance(names, list)
        assert len(names) == num_examples
        names = tf.constant(names)[:, None]
        assert names.dtype == tf.string
        ids = [str(i) for i in range(1, num_examples + 1)]
        ids = tf.constant(ids)[:, None]
        if precision is None:
            if tf_array.dtype in (tf.float32, tf.float64):
                precision = 4
            else:
                precision = -1
        tf_array = tf.as_string(tf_array, precision=precision)
        tf_array = tf.concat([ids, names, tf_array], axis=1)
        tf_array.set_shape([num_examples, num_fields + 2])
        tf_array = tf.strings.reduce_join(tf_array, axis=1, separator=' | ')
        tf_array = tf.strings.reduce_join(tf_array, separator='\n')
        header = tf.strings.reduce_join(header, separator=' | ')
        sep = tf.constant(['---'])
        sep = tf.tile(sep, [num_fields + 2])
        sep = tf.strings.reduce_join(sep, separator=' | ')
        tf_array = tf.strings.join([header, sep, tf_array], separator='\n')
        assert isinstance(title, str)
        tf_array = tf.strings.join([tf.constant(title), tf_array], separator='\n\n')
        assert isinstance(scope, str)
        op = tf.summary.text(scope, tf_array)

        return op

    @staticmethod
    def frame_detector_fn(spec_batch, is_training):

        assert tf.get_variable_scope().name == ''
        spec_batch.set_shape([None, None, 440, 6])
        outputs = spec_batch
        assert isinstance(is_training, bool)

        c_layers = [[128, 5], [64, 5], [64, 3], [64, 3]]
        with tf.variable_scope('frame_detector', reuse=tf.AUTO_REUSE):
            for c_idx, (n_features, k_size) in enumerate(c_layers):
                outputs = slim.conv2d(
                    scope='c_{}'.format(c_idx),
                    inputs=outputs,
                    num_outputs=n_features,
                    kernel_size=k_size,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=dict(is_training=is_training)
                )

            outputs = slim.conv2d(
                scope='c_4',
                inputs=outputs,
                num_outputs=8,
                kernel_size=[3, 70],
                normalizer_fn=slim.batch_norm,
                normalizer_params=dict(is_training=is_training)
            )

            outputs = slim.fully_connected(
                scope='fc_5',
                inputs=outputs,
                num_outputs=1,
                activation_fn=None
            )
            outputs.set_shape([None, None, 440, 1])

            outputs = slim.max_pool2d(
                scope='mp_5',
                inputs=outputs,
                kernel_size=[1, 5],
                stride=[1, 5]
            )
            outputs.set_shape([None, None, 88, 1])
            outputs = tf.squeeze(outputs, axis=-1)
            outputs.set_shape([None, None, 88])

            return outputs

    @staticmethod
    def cal_tps_fps_tns_fns_tf_fn(pred, target):
        assert pred.dtype == tf.bool and target.dtype == tf.bool
        npred = tf.logical_not(pred)
        ntarget = tf.logical_not(target)
        tps = tf.logical_and(pred, target)
        fps = tf.logical_and(pred, ntarget)
        tns = tf.logical_and(npred, ntarget)
        fns = tf.logical_and(npred, target)
        tps, fps, tns, fns = [tf.reduce_sum(tf.cast(value, tf.int32)) for value in [tps, fps, tns, fns]]
        inc_tps_fps_tns_fns = tf.convert_to_tensor([tps, fps, tns, fns], dtype=tf.int64)

        return inc_tps_fps_tns_fns

    @staticmethod
    def to_db_scale_fn(sg):
        return 20. * np.log10(sg + 1e-10) + 200.

    @staticmethod
    def maps_sg_and_label_fn(wav_file):
        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 44100
        num_frames = MiscFns.num_samples_to_num_frames_fn(wav_info.frames)

        rec_name = os.path.basename(wav_file)[:-4]
        hcqt_file = os.path.join(os.environ['maps_hcqt'], rec_name + '.hcqt')
        _rec_name, hcqt = MiscFns.load_np_array_from_file_fn(hcqt_file)
        assert _rec_name == rec_name
        _num_frames = hcqt.shape[0]
        assert _num_frames == num_frames or _num_frames == num_frames + 1
        if _num_frames > num_frames:
            hcqt = hcqt[1:]
        assert hcqt.shape == (num_frames, 440, 6) and hcqt.dtype == np.float32

        mid_file = wav_file[:-3] + 'mid'
        num_frames_from_midi = mido.MidiFile(mid_file).length
        num_frames_from_midi = int(np.ceil(num_frames_from_midi * wav_info.samplerate))
        num_frames_from_midi = MiscFns.num_samples_to_num_frames_fn(num_frames_from_midi)
        num_frames_from_midi += 2
        num_frames = min(num_frames, num_frames_from_midi)
        hcqt = hcqt[:num_frames]

        label = MiscFns.label_fn(mid_file_name=mid_file, num_frames=num_frames)

        hcqt = np.require(hcqt, dtype=np.float32, requirements=['O', 'C'])
        hcqt.flags['WRITEABLE'] = False
        label.flags['WRITEABLE'] = False

        return dict(sg=hcqt, label=label)

    @staticmethod
    def cal_prf_tf_fn(tps, fps, fns):
        assert tps.dtype == tf.float64
        p = tps / (tps + fps + 1e-7)
        r = tps / (tps + fns + 1e-7)
        f = 2. * p * r / (p + r + 1e-7)
        return p, r, f

    @staticmethod
    def get_note_seq_from_mid_file_fn(mid_file_name):
        note_seq = magenta.music.midi_file_to_note_sequence(mid_file_name)
        note_seq = magenta.music.apply_sustain_control_changes(note_seq)

        return note_seq


class Config(object):

    def __init__(self):
        self.debug_mode = DEBUG
        self.gpu_id = GPU_ID
        self.model_names = ['training', 'test', 'validation']

        self.snippet_len = 1200
        self.num_epochs = 25
        self.batches_per_epoch = 5000

        self.learning_rate = 1e-4

        self.train_or_inference = Namespace(
            inference=None,
            from_saved=None,
            model_prefix='d0'
        )
        self.tb_dir = 'tb_d0'

        if not self.debug_mode:
            # check if tb_dir exists
            assert self.tb_dir is not None
            tmp_dirs = glob.glob('./*/')
            tmp_dirs = [s[2:-1] for s in tmp_dirs]
            assert self.tb_dir not in tmp_dirs

            # check if model exists
            if self.train_or_inference.inference is None and self.train_or_inference.model_prefix is not None:
                if os.path.isdir('./saved_model'):
                    tmp_prefixes = glob.glob('./saved_model/*')
                    prog = re.compile(r'./saved_model/(.+?)_')
                    tmp = []
                    for file_name in tmp_prefixes:
                        try:
                            prefix = prog.match(file_name).group(1)
                        except AttributeError:
                            pass
                        else:
                            tmp.append(prefix)
                    tmp_prefixes = set(tmp)
                    assert self.train_or_inference.model_prefix not in tmp_prefixes

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.gpu_config = gpu_config

        self.tvt_split_dict = MiscFns.split_train_valid_and_test_files_fn()

        if self.debug_mode:
            np.random.seed(100)
            for tvt in self.tvt_split_dict.keys():
                _num = len(self.tvt_split_dict[tvt])
                _sel = np.random.choice(_num, 5, replace=False)
                self.tvt_split_dict[tvt] = [self.tvt_split_dict[tvt][ii] for ii in _sel]

            self.num_epochs = 4
            self.batches_per_epoch = 50

        if self.train_or_inference.inference is not None:
            for model_name in self.model_names:
                if model_name in ('training', 'validation'):
                    del self.tvt_split_dict[model_name][1:]


class Model(object):
    def __init__(self, config, name):
        assert name in config.model_names
        self.name = name
        self.is_training = True if self.name == 'training' else False
        self.config = config
        self._gen_dataset_fn()
        self.batch = self._gen_batch_fn()

        with tf.name_scope(self.name):
            logits = self._nn_model_fn()
            logits.set_shape([1, None, 88])
            self.logits = logits
            self.loss = self._loss_fn()
            if self.is_training:
                self.training_op = self._training_op_fn()
            self.stats = self._stats_fn()
            self.tb_proto = self._tb_summary_fn()

    def _dataset_iter_fn(self):
        if self.is_training:
            assert hasattr(self, 'dataset')
            _n_iters = 0
            while True:
                np.random.shuffle(self.rec_start_end_list)
                for rec_idx, start_frame, end_frame in self.rec_start_end_list:
                    rec_dict = self.dataset[rec_idx]
                    yield dict(
                        spectrogram=rec_dict['sg'][start_frame:end_frame],
                        label=rec_dict['label'][start_frame:end_frame],
                        num_frames=end_frame - start_frame
                    )
                _n_iters += 1
                logging.info('{} iterations over the training split done'.format(_n_iters))

        if not self.is_training:
            assert hasattr(self, 'dataset')
            for rec_idx, rec_dict in enumerate(self.dataset):
                split_list = rec_dict['split_list']
                for start_frame, end_frame in split_list:
                    yield dict(
                        spectrogram=rec_dict['sg'][start_frame:end_frame],
                        label=rec_dict['label'][start_frame:end_frame],
                        num_frames=end_frame - start_frame,
                        rec_idx=rec_idx
                    )

    def _gen_batch_fn(self):

        _sg_shape = [None, 440, 6]
        with tf.device('/cpu:0'):
            if self.is_training:
                dataset = tf.data.Dataset.from_generator(
                    generator=self._dataset_iter_fn,
                    output_types=dict(spectrogram=tf.float32, label=tf.bool, num_frames=tf.int32),
                    output_shapes=dict(spectrogram=_sg_shape, label=[None, 88], num_frames=[])
                )
                dataset = dataset.batch(1)

                dataset = dataset.prefetch(50)
                dataset_iter = dataset.make_one_shot_iterator()
                element = dataset_iter.get_next()

                return element
            else:  # not self.is_training
                dataset = tf.data.Dataset.from_generator(
                    generator=self._dataset_iter_fn,
                    output_types=dict(spectrogram=tf.float32, label=tf.bool, num_frames=tf.int32, rec_idx=tf.int32),
                    output_shapes=dict(spectrogram=_sg_shape, label=[None, 88], num_frames=[], rec_idx=[])
                )
                dataset = dataset.batch(1)
                dataset = dataset.prefetch(50)
                self.reinitializable_iter_for_dataset = dataset.make_initializable_iterator()
                element = self.reinitializable_iter_for_dataset.get_next()
                element['spectrogram'].set_shape([1] + _sg_shape)
                element['label'].set_shape([1, None, 88])
                element['num_frames'].set_shape([1])
                element['rec_idx'].set_shape([1])

                return element

    def _nn_model_fn(self):
        inputs = self.batch['spectrogram']
        _nn_fn = MiscFns.frame_detector_fn
        inputs.set_shape([1, None, 440, 6])

        outputs = _nn_fn(spec_batch=inputs, is_training=self.is_training)
        outputs.set_shape([1, None, 88])

        return outputs

    def _gen_dataset_fn(self):

        assert not hasattr(self, 'dataset')

        if self.is_training:
            file_names = self.config.tvt_split_dict[self.name]
            num_recs = len(file_names)
            dataset = []

            for file_idx, wav_file_name in enumerate(file_names):
                rec_name = os.path.basename(wav_file_name)[:-4]
                logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, rec_name))
                sg_label_dict = MiscFns.maps_sg_and_label_fn(wav_file_name)
                dataset.append(sg_label_dict)
            self.dataset = dataset

            rec_start_end_list = []
            for rec_idx, rec_dict in enumerate(self.dataset):
                split_list = MiscFns.gen_split_list_fn(
                    num_frames=len(rec_dict['sg']), snippet_len=self.config.snippet_len
                )
                tmp = [[rec_idx] + se for se in split_list]
                rec_start_end_list.extend(tmp)
            self.rec_start_end_list = rec_start_end_list
            logging.info('num of snippets per iteration over the dataset - {}'.format(len(self.rec_start_end_list)))

        if not self.is_training:
            file_names = self.config.tvt_split_dict[self.name]
            num_recs = len(file_names)
            dataset = []
            rec_names = []
            for file_idx, wav_file_name in enumerate(file_names):
                rec_name = os.path.basename(wav_file_name)[:-4]
                rec_names.append(rec_name)
                logging.info('{}/{} - {}'.format(file_idx + 1, num_recs, rec_name))
                sg_label_dict = MiscFns.maps_sg_and_label_fn(wav_file_name)
                split_list = MiscFns.gen_split_list_fn(
                    num_frames=len(sg_label_dict['sg']), snippet_len=self.config.snippet_len)
                sg_label_dict['split_list'] = split_list

                dataset.append(sg_label_dict)
            self.dataset = dataset
            self.rec_names = tuple(rec_names)

            self.num_frames_vector = np.asarray([len(rec_dict['sg']) for rec_dict in self.dataset], dtype=np.int64)

    def _loss_fn(self):

        logits = self.logits
        logits.set_shape([1, None, 88])
        logits = tf.squeeze(logits, axis=0)
        labels = self.batch['label']
        labels.set_shape([1, None, 88])
        assert labels.dtype == tf.bool
        labels = tf.squeeze(labels, axis=0)
        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)

        return loss

    def _training_op_fn(self):
        assert self.is_training
        loss = self.loss
        _update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if _update_ops:
            with tf.control_dependencies(_update_ops):
                training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
        else:
            training_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)

        return training_op

    def _stats_fn(self):

        if not self.is_training:
            assert tf.get_variable_scope().name == ''
            num_recs = len(self.dataset)
            with tf.variable_scope(self.name):
                with tf.variable_scope('statistics'):
                    var_int64_ind_tps_fps_tns_fns = tf.get_variable(
                        name='var_int64_ind_tps_fps_tns_fns',
                        dtype=tf.int64,
                        shape=[num_recs, 4],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )
                    var_float64_acc_loss = tf.get_variable(
                        name='var_float64_acc_loss',
                        dtype=tf.float64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )
                    var_int64_batch_counter = tf.get_variable(
                        name='var_int64_batch_counter',
                        dtype=tf.int64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    pred_labels = tf.greater(tf.squeeze(self.logits, axis=0), 0.)
                    pred_labels.set_shape([None, 88])
                    target_labels = self.batch['label']
                    assert target_labels.dtype == tf.bool
                    target_labels.set_shape([1, None, 88])
                    target_labels = tf.squeeze(target_labels, axis=0)
                    inc = MiscFns.cal_tps_fps_tns_fns_tf_fn(
                        pred=pred_labels,
                        target=target_labels
                    )
                    assert inc.dtype == tf.int64
                    assert inc._rank() == 1

                    num_labels = tf.cast(self.batch['num_frames'][0], tf.int64) * tf.constant(88, dtype=tf.int64)
                    _num_labels = tf.reduce_sum(inc)
                    _assert_op = tf.assert_equal(num_labels, _num_labels)
                    with tf.control_dependencies([_assert_op]):
                        ind_update_op = tf.scatter_add(
                            var_int64_ind_tps_fps_tns_fns,
                            self.batch['rec_idx'][0],
                            inc
                        )

                    acc_loss_update_op = tf.assign_add(var_float64_acc_loss, tf.cast(self.loss, tf.float64))
                    batch_counter_update_op = tf.assign_add(var_int64_batch_counter, tf.constant(1, dtype=tf.int64))

                    # ind and average stats
                    tps, fps, _, fns = tf.unstack(tf.cast(var_int64_ind_tps_fps_tns_fns, tf.float64), axis=1)
                    ps, rs, fs = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ind_prfs = tf.stack([ps, rs, fs], axis=1)
                    ind_prfs.set_shape([num_recs, 3])
                    _num_labels = self.num_frames_vector
                    assert isinstance(_num_labels, np.ndarray) and _num_labels.dtype == np.int64
                    _num_labels = tf.constant(_num_labels) * tf.constant(88, dtype=tf.int64)
                    _num_labels_p = tf.reduce_sum(var_int64_ind_tps_fps_tns_fns, axis=1)
                    _assert_op = tf.assert_equal(_num_labels, _num_labels_p)
                    with tf.control_dependencies([_assert_op]):
                        ave_prf = tf.reduce_mean(ind_prfs, axis=0)

                    # ensemble stats
                    ensemble = tf.reduce_sum(var_int64_ind_tps_fps_tns_fns, axis=0)
                    tps, fps, _, fns = tf.unstack(tf.cast(ensemble, tf.float64))
                    p, r, f = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ave_loss = var_float64_acc_loss / tf.cast(var_int64_batch_counter, tf.float64)
                    with tf.control_dependencies([_assert_op]):
                        en_prf_and_loss = tf.convert_to_tensor([p, r, f, ave_loss])

                    update_op = tf.group(ind_update_op, batch_counter_update_op, acc_loss_update_op)

                    stats = dict(
                        individual_tps_fps_tns_fns=var_int64_ind_tps_fps_tns_fns,
                        individual_prfs=ind_prfs,
                        average_prf=ave_prf,
                        ensemble_tps_fps_tns_fns=ensemble,
                        ensemble_prf_and_loss=en_prf_and_loss
                    )
            return dict(update_op=update_op, value=stats)

        if self.is_training:
            logits = self.logits
            logits.set_shape([1, None, 88])
            loss = self.loss
            with tf.variable_scope(self.name):
                with tf.variable_scope('statistics'):
                    var_int64_tps_fps_tns_fns = tf.get_variable(
                        name='var_int64_tps_fps_tns_fns',
                        dtype=tf.int64,
                        shape=[4],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    var_float64_acc_loss = tf.get_variable(
                        name='var_float64_acc_loss',
                        dtype=tf.float64,
                        shape=[],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    var_int64_batch_counter = tf.get_variable(
                        name='var_int64_batch_counter',
                        dtype=tf.int64,
                        shape=[],
                        trainable=False,
                        initializer=tf.zeros_initializer,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES]
                    )

                    pred_labels_flattened = tf.greater(tf.squeeze(logits, axis=0), 0.)

                    target_labels_flattened = self.batch['label']
                    target_labels_flattened = tf.squeeze(target_labels_flattened, axis=0)
                    inc_tps_fps_tns_fns = MiscFns.cal_tps_fps_tns_fns_tf_fn(
                        pred=pred_labels_flattened,
                        target=target_labels_flattened
                    )
                    assert inc_tps_fps_tns_fns.dtype == tf.int64
                    num_labels = self.batch['num_frames'][0]
                    num_labels = tf.cast(num_labels, tf.int64) * tf.constant(88, tf.int64)
                    _num_labels = tf.reduce_sum(inc_tps_fps_tns_fns)
                    _assert_op = tf.assert_equal(num_labels, _num_labels)
                    with tf.control_dependencies([_assert_op]):
                        ensemble_tps_fps_tns_fns_update_op = tf.assign_add(
                            var_int64_tps_fps_tns_fns,
                            inc_tps_fps_tns_fns
                        )

                    acc_loss_update_op = tf.assign_add(var_float64_acc_loss, tf.cast(loss, tf.float64))
                    batch_counter_update_op = tf.assign_add(var_int64_batch_counter, tf.constant(1, tf.int64))

                    en_tps_fps_tns_fns_float64 = tf.cast(var_int64_tps_fps_tns_fns, tf.float64)
                    tps, fps, _, fns = tf.unstack(en_tps_fps_tns_fns_float64)
                    p, r, f = MiscFns.cal_prf_tf_fn(tps=tps, fps=fps, fns=fns)
                    ave_loss = var_float64_acc_loss / tf.cast(var_int64_batch_counter, tf.float64)
                    prf_and_ave_loss = tf.convert_to_tensor([p, r, f, ave_loss])

                    update_op = tf.group(ensemble_tps_fps_tns_fns_update_op, acc_loss_update_op, batch_counter_update_op)

            return dict(update_op=update_op, value=prf_and_ave_loss)

    def _tb_summary_fn(self):

        if self.is_training:
            scalar_summaries = []
            with tf.name_scope('statistics'):
                p, r, f, l = tf.unstack(self.stats['value'])
                summary_dict = dict(
                    precision=p,
                    recall=r,
                    f1=f,
                    loss=l
                )

                for sum_name, sum_value in summary_dict.iteritems():
                    scalar_summaries.append(tf.summary.scalar(sum_name, sum_value))
                scalar_summaries = tf.summary.merge(scalar_summaries)

            return scalar_summaries

        if not self.is_training:
            num_recs = len(self.dataset)
            tb_table_and_scalar_protos = []
            with tf.name_scope('statistics'):
                stats = self.stats['value']
                ind_prfs = stats['individual_prfs']
                ave_prf = stats['average_prf']
                en_prf, ave_loss = tf.split(stats['ensemble_prf_and_loss'], [3, 1])
                ave_loss = ave_loss[0]

                assert ind_prfs.dtype == ave_prf.dtype == en_prf.dtype == tf.float64
                prfs = tf.concat([ind_prfs, ave_prf[None, :], en_prf[None, :]], axis=0)
                prfs.set_shape([num_recs + 2, 3])
                names = list(self.rec_names) + ['average', 'ensemble']
                des = 'individual_and_their_average_and_ensemble_prfs'
                prf_tb_table_proto = MiscFns.array_to_table_tf_fn(
                    tf_array=prfs,
                    header=['p', 'r', 'f1'],
                    scope=des,
                    title=des.replace('_', ' '),
                    names=names
                )
                tb_table_and_scalar_protos.append(prf_tb_table_proto)

                ind_tps_fps_tns_fns = stats['individual_tps_fps_tns_fns']
                en_tps_fps_tns_fns = stats['ensemble_tps_fps_tns_fns']
                en_tps_fps_tns_fns = en_tps_fps_tns_fns[None, :]
                assert ind_tps_fps_tns_fns.dtype.base_dtype == en_tps_fps_tns_fns.dtype == tf.int64
                ind_en = tf.concat([ind_tps_fps_tns_fns, en_tps_fps_tns_fns], axis=0)
                names = list(self.rec_names)
                names.append('ensemble')
                des = 'individual and their ensemble tps fps tns and fns'
                tps_fps_tns_fns_tb_table_proto = MiscFns.array_to_table_tf_fn(
                    tf_array=ind_en,
                    header=['TP', 'FP', 'TN', 'FN'],
                    scope=des.replace(' ', '_'),
                    title=des,
                    names=names
                )
                tb_table_and_scalar_protos.append(tps_fps_tns_fns_tb_table_proto)

                prf_loss_tb_scalar_proto = []
                for ave_or_en in ('average', 'ensemble'):
                    with tf.name_scope(ave_or_en):
                        p, r, f = tf.unstack(ave_prf if ave_or_en == 'average' else en_prf)
                        prfl_protos = []
                        items_for_summary = dict(precision=p, recall=r, f1=f)
                        for item_name, item_value in items_for_summary.iteritems():
                            prfl_protos.append(tf.summary.scalar(item_name, item_value))
                        if ave_or_en == 'average':
                            prfl_protos.append(tf.summary.scalar('loss', ave_loss))
                        prf_loss_tb_scalar_proto.append(tf.summary.merge(prfl_protos))
                prf_loss_tb_scalar_proto = tf.summary.merge(prf_loss_tb_scalar_proto)

                tb_table_and_scalar_protos.append(prf_loss_tb_scalar_proto)
                tb_table_and_scalar_protos = tf.summary.merge(tb_table_and_scalar_protos)

            return tb_table_and_scalar_protos


def main():
    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info_pl = tf.placeholder(dtype=tf.string, name='aug_info_pl')
    aug_info_summary = tf.summary.text('aug_info_summary', aug_info_pl)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(MODEL_DICT['config'].gpu_id)
    with tf.Session(config=MODEL_DICT['config'].gpu_config) as sess:
        # summary writer
        summary_writer_dict = {}
        for model_name in MODEL_DICT['config'].model_names:
            summary_writer_dict[model_name] = tf.summary.FileWriter(
                os.path.join(MODEL_DICT['config'].tb_dir, model_name))

        aug_info = []
        aug_info.append('hcqt')

        if MODEL_DICT['config'].train_or_inference.inference is not None:
            aug_info.append('inference with {}'.format(MODEL_DICT['config'].train_or_inference.inference))
        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            aug_info.append('continue training from {}'.format(MODEL_DICT['config'].train_or_inference.from_saved))

        if MODEL_DICT['config'].train_or_inference.inference is None:
            _model_prefix = MODEL_DICT['config'].train_or_inference.model_prefix
            if _model_prefix is not None:
                aug_info.append('model prefix {}'.format(_model_prefix))
            else:
                aug_info.append('model will not be saved')

        aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
        aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
        aug_info.append('snippet length - {}'.format(MODEL_DICT['config'].snippet_len))
        aug_info.append('batch size - 1')
        aug_info.append('num of batches per epoch - {}'.format(MODEL_DICT['config'].batches_per_epoch))
        aug_info.append('num of epochs - {}'.format(MODEL_DICT['config'].num_epochs))
        aug_info.append('training start time - {}'.format(datetime.datetime.now()))
        aug_info = '\n\n'.join(aug_info)
        logging.info(aug_info)
        summary_writer_dict[MODEL_DICT['config'].model_names[0]].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: aug_info}))

        logging.info('local vars -')
        for idx, var in enumerate(tf.local_variables()):
            logging.info('{}\t{}'.format(idx, var.op.name))

        logging.info('trainable vars -')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
            logging.info('{}\t{}\t{}'.format(idx, var.op.name, var.shape))

        OP_DICT = {}
        for model_name in MODEL_DICT['config'].model_names:
            m = MODEL_DICT[model_name]

            if m.is_training:
                tmp = dict(
                    batch=[m.training_op, m.stats['update_op']],
                    epoch=m.tb_proto
                )
            else:
                tmp = dict(
                    batch=dict(
                        rec_idx=m.batch['rec_idx'],
                        update_op=m.stats['update_op']
                    ),
                    epoch=m.tb_proto
                )

            OP_DICT[model_name] = tmp

        def test_or_validate_fn(valid_or_test, global_step=None):
            assert valid_or_test in MODEL_DICT['config'].model_names and 'training' not in valid_or_test

            ops_per_batch = OP_DICT[valid_or_test]['batch']
            ops_per_epoch = OP_DICT[valid_or_test]['epoch']

            batch_idx = 0
            _dataset_test = MODEL_DICT[valid_or_test].dataset
            total_num_snippets = sum(len(rec_dict['split_list']) for rec_dict in _dataset_test)
            num_recs = len(_dataset_test)

            for rec_idx in xrange(num_recs):
                rec_dict = _dataset_test[rec_idx]
                split_list = rec_dict['split_list']
                num_snippets = len(split_list)
                num_frames = len(rec_dict['sg'])
                assert num_frames == MODEL_DICT[valid_or_test].num_frames_vector[rec_idx]
                for snippet_idx in xrange(num_snippets):
                    logging.debug('batch {}/{}'.format(batch_idx + 1, total_num_snippets))
                    tmp = sess.run(ops_per_batch)
                    _rec_idx = tmp['rec_idx'][0]
                    assert _rec_idx == rec_idx
                    batch_idx += 1
            summary_writer_dict[valid_or_test].add_summary(sess.run(ops_per_epoch), global_step)

        def check_all_global_vars_initialized_fn():
            tmp = sess.run(tf.report_uninitialized_variables(tf.global_variables()))
            assert tmp.size == 0

        if MODEL_DICT['config'].train_or_inference.inference is not None:
            save_path = MODEL_DICT['config'].train_or_inference.inference
            tf.train.Saver().restore(sess, save_path)
            check_all_global_vars_initialized_fn()

            sess.run(tf.initializers.variables(tf.local_variables()))
            for model_name in MODEL_DICT['config'].model_names:
                if 'test' in model_name:
                    sess.run(MODEL_DICT[model_name].reinitializable_iter_for_dataset.initializer)
                    logging.info('do inference on {}'.format(model_name))
                    test_or_validate_fn(model_name)

        elif MODEL_DICT['config'].train_or_inference.from_saved is not None:
            save_path = MODEL_DICT['config'].train_or_inference.from_saved
            tf.train.Saver().restore(sess, save_path)
            check_all_global_vars_initialized_fn()

            logging.info('reproduce results ...')
            sess.run(tf.initializers.variables(tf.local_variables()))
            for model_name in MODEL_DICT['config'].model_names:
                if 'training' not in model_name:
                    sess.run(MODEL_DICT[model_name].reinitializable_iter_for_dataset.initializer)

            for model_name in MODEL_DICT['config'].model_names:
                if 'training' not in model_name:
                    logging.info(model_name)
                    test_or_validate_fn(model_name, 0)

        else:  # neither inference or from saved
            logging.info('train from scratch')
            sess.run(tf.initializers.variables(tf.global_variables()))
            check_all_global_vars_initialized_fn()

        if MODEL_DICT['config'].train_or_inference.inference is None:
            check_all_global_vars_initialized_fn()
            if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                assert 'model_saver' not in MODEL_DICT
                MODEL_DICT['model_saver'] = tf.train.Saver(max_to_keep=200)

            for training_valid_test_epoch_idx in xrange(MODEL_DICT['config'].num_epochs):
                logging.info('\n\ncycle - {}/{}'.format(training_valid_test_epoch_idx + 1, MODEL_DICT['config'].num_epochs))

                sess.run(tf.initializers.variables(tf.local_variables()))

                # to enable prefetch
                for model_name in MODEL_DICT['config'].model_names:
                    if 'training' not in model_name:
                        sess.run(MODEL_DICT[model_name].reinitializable_iter_for_dataset.initializer)

                for training_valid_or_test in MODEL_DICT['config'].model_names:
                    logging.info(training_valid_or_test)

                    if 'training' in training_valid_or_test:
                        ops_per_batch = OP_DICT[training_valid_or_test]['batch']
                        ops_per_epoch = OP_DICT[training_valid_or_test]['epoch']
                        for batch_idx in xrange(MODEL_DICT['config'].batches_per_epoch):
                            sess.run(ops_per_batch)
                            logging.debug('batch - {}/{}'.format(batch_idx + 1, MODEL_DICT['config'].batches_per_epoch))
                        summary_writer_dict[training_valid_or_test].add_summary(
                            sess.run(ops_per_epoch),
                            training_valid_test_epoch_idx + 1
                        )

                        if MODEL_DICT['config'].train_or_inference.model_prefix is not None:
                            save_path = MODEL_DICT['config'].train_or_inference.model_prefix + \
                                        '_' + 'epoch_{}_of_{}'.format(training_valid_test_epoch_idx + 1,
                                                                      MODEL_DICT['config'].num_epochs)
                            save_path = os.path.join('saved_model', save_path)
                            save_path = MODEL_DICT['model_saver'].save(
                                sess=sess,
                                save_path=save_path,
                                global_step=None,
                                write_meta_graph=False
                            )
                            logging.info('model saved to {}'.format(save_path))

                    else:
                        test_or_validate_fn(training_valid_or_test, training_valid_test_epoch_idx + 1)

        msg = 'training end time - {}'.format(datetime.datetime.now())
        logging.info(msg)
        summary_writer_dict[MODEL_DICT['config'].model_names[0]].add_summary(sess.run(aug_info_summary, feed_dict={aug_info_pl: msg}))

        for training_valid_or_test in MODEL_DICT['config'].model_names:
            summary_writer_dict[training_valid_or_test].close()


def debug_dataset_iter_fn():
    config = Config()
    model_training = Model(config, 'training')
    dit = model_training._dataset_iter_fn()
    ele = dit.next()
    model_test = Model(config, 'test')
    dit = model_test._dataset_iter_fn()
    ele = dit.next()


if __name__ == '__main__':
   main()

















