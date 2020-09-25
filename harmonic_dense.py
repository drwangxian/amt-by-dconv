@add_arg_scope
def harmonic_dense(
        inputs,
        harmonic_bins,
        num_outputs,
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        reuse=None,
        variable_collections=None,
        weights_regularizer=None,
        outputs_collections=None,
        trainable=True,
        pitch_bins=range(88 * 3),
        scope=None
):
    num_pitches = len(pitch_bins)
    num_freq_bins = inputs.shape[-2].value
    assert num_freq_bins is not None
    assert num_freq_bins >= num_pitches
    num_input_features = inputs.shape[-1].value
    assert num_input_features is not None
    num_output_features = num_outputs
    num_harmonics = len(harmonic_bins)

    harmonic_bins = np.asarray(harmonic_bins, dtype=np.int32)
    assert harmonic_bins.ndim == 1
    harmonic_matrix = []
    for pitch in pitch_bins:
        assert 0 <= pitch < num_freq_bins
        harmonic_matrix.append(harmonic_bins + pitch)
    harmonic_matrix = np.stack(harmonic_matrix, axis=0)
    assert harmonic_matrix.shape == (num_pitches, num_harmonics)
    fan_in = np.logical_and(harmonic_matrix >= 0, harmonic_matrix < num_freq_bins)
    fan_in = 1. * np.sum(fan_in) * num_input_features / num_pitches

    fan_out = np.zeros((num_freq_bins, num_input_features), dtype=np.int64)
    for idx, pitch in enumerate(pitch_bins):
        harmonic_bin_vector = harmonic_matrix[idx]
        for harmonic in harmonic_bin_vector:
            if 0 <= harmonic < num_freq_bins:
                fan_out[harmonic] += num_output_features
    fan_out = np.mean(fan_out)
    uni_bound = np.sqrt(12. / (fan_in + fan_out))
    weights_initializer = init_ops.random_uniform_initializer(minval=-uni_bound, maxval=uni_bound)

    with variable_scope.variable_scope(scope, 'HarmonicDense', [inputs], reuse=reuse) as sc:
        weights = variables.model_variable(
            name='weights',
            shape=[num_input_features, num_harmonics, num_output_features],
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable,
            collections=None
        )
        _add_variable_to_collections(weights, variable_collections, 'weights')
        if normalizer_fn is None:
            biases = variables.model_variable(
                name='biases',
                shape=[num_output_features],
                initializer=init_ops.zeros_initializer(),
                trainable=trainable,
                collections=None
            )
            _add_variable_to_collections(biases, variable_collections, 'biases')

        weights = weights[None, ...]

        outputs_for_all_pitches = []
        for pidx, pitch in enumerate(pitch_bins):
            harmonic_bin_vector = harmonic_matrix[pidx]
            valid_harmonic_idx = np.where(np.logical_and(harmonic_bin_vector >= 0, harmonic_bin_vector < num_freq_bins))[0]
            harmonic_batch = array_ops.gather(
                params=inputs,
                indices=array_ops.constant(harmonic_bin_vector[valid_harmonic_idx], dtype=dtypes.int32),
                axis=2
            )
            harmonic_batch.set_shape([None, None, len(valid_harmonic_idx), num_input_features])
            harmonic_batch = array_ops.transpose(harmonic_batch, perm=[0, 1, 3, 2])
            weights_for_this_pitch = array_ops.gather(
                weights, indices=array_ops.constant(valid_harmonic_idx, dtype=dtypes.int32), axis=2)
            weights_for_this_pitch.set_shape([1, num_input_features, len(valid_harmonic_idx), num_output_features])
            outputs_for_this_pitch = nn.conv2d(
                input=harmonic_batch,
                filter=weights_for_this_pitch,
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            outputs_for_this_pitch.set_shape([None, None, 1, num_output_features])
            outputs_for_all_pitches.append(outputs_for_this_pitch)

        outputs = array_ops.concat(outputs_for_all_pitches, axis=2)
        outputs.set_shape([None, None, num_pitches, num_output_features])

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            outputs = outputs + biases

        if activation_fn  is not None:
            outputs = activation_fn(outputs)

        outputs = utils.collect_named_outputs(outputs_collections, sc.name, outputs)

    return outputs