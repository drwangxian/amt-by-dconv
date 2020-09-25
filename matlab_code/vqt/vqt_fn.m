function [c_AC_ensemble, err_db] = vqt_fn(varargin)
    
%{
    db_scale: true (default)  or false
    mono: true (default) or false
    sub_sampling_factor: int, default to 22
    gamma: double, default to 14.112 Hz (equivalent to 9000 samples at sr of 44100) 
    wav_file: string
    gpu_idx: int, when gpu_idx >=1, do fft on gpu. Be advised that in Matlab, gpu index starts from 1.
    %}

    % read variable input arguments, if not present, set to default value
    nvarargin = length(varargin);
    assert(mod(nvarargin, 2) == 0);
   
    for vid=1:2:nvarargin
        key = varargin{vid};
        value = varargin{vid + 1};
        assert(ischar(key));
        switch key
            case 'db_scale'
                db_scale = value;
            case 'mono'
                mono = value;
            case 'sub_sampling_factor'
                sub_sampling_factor = value;
            case 'gamma'
                gamma = value;
            case 'wav_file'
                wav_file = value;
            case 'gpu_idx'
                gpu_idx = value;
            otherwise
                error('unknown key %s', key);
        end
    end
    
    if exist('db_scale', 'var') == 0
        db_scale = true;
    end
    
    if exist('mono', 'var') == 0
        mono = true;
    end
    
    if exist('sub_sampling_factor', 'var') == 0
        sub_sampling_factor = 22;
    end
    
    if exist('gamma', 'var') == 0
        gamma = 14.112;
    end
    
    assert(exist('wav_file', 'var') == 1);
    
    if exist('gpu_idx', 'var') == 0
        gpu_idx = -1;
    end
    
    %check validity of parameters
    assert(isscalar(db_scale) && islogical(db_scale));
    assert(isscalar(mono) && islogical(mono));
    
    assert(isscalar(sub_sampling_factor));
    assert(isa(sub_sampling_factor, 'int64') || isa(sub_sampling_factor, 'double'));
    if isa(sub_sampling_factor, 'int64')
        sub_sampling_factor = double(sub_sampling_factor);
    end
    assert(mod(sub_sampling_factor, 1) == 0);
    
    assert(isscalar(gamma));
    assert(isa(gamma, 'int64') || isa(gamma, 'double'));
    if isa(gamma, 'int64')
       gamma = double(gamma); 
    end
    
    assert(isstring(wav_file) || ischar(wav_file));
    
    assert(isscalar(gpu_idx));
    assert(isa(gpu_idx, 'int64') || isa(gpu_idx, 'double'));
    if isa(gpu_idx, 'int64')
        gpu_idx = double(gpu_idx);
    end
    assert(mod(gpu_idx, 1) == 0);
    
    assert(nargout == 1 || nargout == 2);
    
    
    partition_threshold = 1300;  % seconds
    
    if nargout == 2
        fprintf('\nsetting:\n');
        fprintf('dB scale - %d\n', db_scale);
        fprintf('mono - %d\n', mono);
        fprintf('sub-sampling factor - %d\n', sub_sampling_factor);
        fprintf('gamma - %.2f\n', gamma); 
        fprintf('gpu_idx - %d\n', gpu_idx);
        fprintf('partition threshold - %.0f s\n', partition_threshold);
        disp('');
    end
    
    sr = 44100;
    hop_size = 64;
    wav_info = audioinfo(wav_file);
    assert(wav_info.SampleRate == 44100 || wav_info.SampleRate == 48000);
    assert(wav_info.BitsPerSample == 16);
    x = audioread(wav_file, 'native');
    assert(ismatrix(x));
    assert(isa(x, 'int16'));
    x = double(x) / 32768;
    assert(size(x, 2) == 1 || size(x, 2) == 2);
    if mono && size(x, 2) == 2
       x = x(:, 1); 
    end
    
    if wav_info.SampleRate > 44100
       x = resample(x, sr, wav_info.SampleRate);
       fprintf('warning: the original sample rate is %d and downsampled to %d\n', wav_info.SampleRate, sr);
    end
    
    num_chs = size(x, 2); 
    if mono
        assert(num_chs == 1); 
    end
    Ls_original = length(x);
    num_hops_original = ceil(Ls_original / hop_size);
    num_hops_original_down_sampled = ceil(num_hops_original / sub_sampling_factor);
    
    num_partitions = ceil(wav_info.Duration / partition_threshold);
    if num_partitions > 1
        fprintf('partition threshold - %.0f s\n', partition_threshold);
        fprintf('duration - %.0f s\n', wav_info.Duration);
        fprintf('num of partitions - %d\n', num_partitions);
    end
    num_hops_per_partition = ceil(ceil(Ls_original / num_partitions) / hop_size);
    total_hops_after_partition = num_hops_per_partition * num_partitions;
    Ls_after_partition = total_hops_after_partition * hop_size;
    assert(total_hops_after_partition >= num_hops_original);
    num_paddings_to_enable_partition = Ls_after_partition - Ls_original;
    if num_paddings_to_enable_partition > 0
        x = padarray(x, num_paddings_to_enable_partition, 'post');
        assert(isequal(size(x), [Ls_after_partition, num_chs]));
    end
    
    assert(mod(Ls_after_partition, num_partitions) == 0);
    Ls_per_partition_before_anti_circular_shift_padding = Ls_after_partition / num_partitions;
    corr_len = 2.88 * sr / gamma;
    num_padded_hops_per_partition = ceil(ceil(corr_len / 2 ) / hop_size);
    num_paddings_per_partition_to_anti_circular_shift = num_padded_hops_per_partition * hop_size;
    Ls_per_partition_after_anti_circular_shift_padding = Ls_per_partition_before_anti_circular_shift_padding + ...
        2 * num_paddings_per_partition_to_anti_circular_shift;
    
    B = 36;
    assert(mod(B, 12) == 0);
    bins_per_note = B / 12;
    assert(mod(bins_per_note, 2) == 1);
    fmin = midi2freq(21) * 2 ^ (-(bins_per_note - 1) / 2 / B);
    fmax = midi2freq(108 + 24) * 2 ^ ((bins_per_note - 1) / 2 / B);
    [fbas, cqtbw] = hd_fbas_cqtbw_fn(fmin, fmax, gamma, B, sr);
    [g, fbas] = nsgcqwin_with_dc_and_nyquist_fn(fbas, cqtbw, sr, Ls_per_partition_after_anti_circular_shift_padding);
    assert(length(g) == (88 + 24) * bins_per_note + 2);
    assert(mod(Ls_per_partition_after_anti_circular_shift_padding, hop_size) == 0);
    M_AC = Ls_per_partition_after_anti_circular_shift_padding / hop_size;
    assert(M_AC >= max(cellfun(@length, g(2:end - 1))));
    M_DC = ceil(length(g{1}) / hop_size) * hop_size;
    M_nyq = ceil(length(g{end}) / hop_size) * hop_size;
    Ms = zeros(size(fbas));
    Ms(1) = M_DC;
    Ms(2:end - 1) = M_AC;
    Ms(end) = M_nyq;
    for idx =1:length(g)
       g{idx} = g{idx} * (2 * Ms(idx) / Ls_per_partition_after_anti_circular_shift_padding); 
    end
    
    if nargout == 2
        gd = nsdual_xian(g, fbas, Ms, Ls_per_partition_after_anti_circular_shift_padding);
        x_recovered = zeros(size(x));
    end
    
    assert(mod(Ls_after_partition, hop_size) == 0);
    num_down_sampled_hops_after_partition = ceil(Ls_after_partition / hop_size / sub_sampling_factor);
    assert(num_down_sampled_hops_after_partition >= num_hops_original_down_sampled);
    num_freq_bins = length(g) - 2;
    c_AC_ensemble = zeros([num_down_sampled_hops_after_partition, num_freq_bins, num_chs], 'single');
  
    for partition_idx=1:num_partitions
        if partition_idx == 1
            partition_start_sample = 1;
            partition_end_sample = Ls_per_partition_before_anti_circular_shift_padding;
            x_partition = x(partition_start_sample:partition_end_sample, :);
            x_partition = padarray(x_partition, num_paddings_per_partition_to_anti_circular_shift, 'pre');
            if num_partitions > 1
                end_padding_start_sample = partition_end_sample + 1;
                end_padding_end_sample = partition_end_sample + num_paddings_per_partition_to_anti_circular_shift;
                assert(end_padding_end_sample < Ls_after_partition);
                end_padding_block = x(end_padding_start_sample:end_padding_end_sample,:);
                x_partition = [x_partition;end_padding_block];
            else
                x_partition = padarray(x_partition, num_paddings_per_partition_to_anti_circular_shift, 'post');
            end
        elseif partition_idx < num_partitions
            partition_start_sample = (partition_idx - 1) * Ls_per_partition_before_anti_circular_shift_padding + 1;
            partition_start_sample = partition_start_sample - num_paddings_per_partition_to_anti_circular_shift;
            assert(partition_start_sample > 0);
            partition_end_sample = partition_idx * Ls_per_partition_before_anti_circular_shift_padding;
            partition_end_sample = partition_end_sample + num_paddings_per_partition_to_anti_circular_shift;
            assert(partition_end_sample < Ls_after_partition);
            x_partition = x(partition_start_sample:partition_end_sample, :);
        else
            partition_start_sample = (partition_idx - 1) * Ls_per_partition_before_anti_circular_shift_padding + 1;
            partition_start_sample = partition_start_sample - num_paddings_per_partition_to_anti_circular_shift;
            assert(partition_start_sample > 0);
            partition_end_sample = partition_idx * Ls_per_partition_before_anti_circular_shift_padding;
            assert(partition_end_sample == Ls_after_partition);
            x_partition = x(partition_start_sample:partition_end_sample, :);
            x_partition = padarray(x_partition, num_paddings_per_partition_to_anti_circular_shift, 'post');
        end
        assert(isequal(size(x_partition), [Ls_per_partition_after_anti_circular_shift_padding, num_chs]));
        
    
        [c_DC, c_AC, c_nyq] = nsgtf_real_xian(x_partition, g, fbas, M_DC, M_AC, M_nyq, 'global', gpu_idx);
        
        if gpu_idx >= 1
           gpuDevice([]); 
        end

        assert(isequal(size(c_AC), [M_AC, num_freq_bins]) || isequal(size(c_AC), [M_AC, num_freq_bins, num_chs]));
        assert(isequal(size(c_DC), [M_DC, num_chs]));
        assert(isequal(size(c_nyq), [M_nyq, num_chs]));

        if nargout == 2
            y_partition = nsigtf_real_xian(c_DC, c_AC, c_nyq, gd, fbas, Ls_per_partition_after_anti_circular_shift_padding, 'global', gpu_idx);
            assert(isequal(size(y_partition), size(x_partition)));
            x_recovered_start_sample = (partition_idx - 1) * Ls_per_partition_before_anti_circular_shift_padding + 1;
            x_recovered_end_sample = partition_idx * Ls_per_partition_before_anti_circular_shift_padding;
            x_recovered(x_recovered_start_sample:x_recovered_end_sample, :) = ...
                y_partition(num_paddings_per_partition_to_anti_circular_shift + 1:end - num_paddings_per_partition_to_anti_circular_shift, :);
        end
        
        if partition_idx == 1
           c_AC_ensemble_start_idx = 1;
           offset_idx_for_c_AC_partition = 0;
        end
        
        num_hops_down_sampled_for_this_partition = ceil((num_hops_per_partition - offset_idx_for_c_AC_partition) / sub_sampling_factor);
        c_AC_ensemble_end_indx = c_AC_ensemble_start_idx + num_hops_down_sampled_for_this_partition - 1;
        if partition_idx == num_partitions
           assert(c_AC_ensemble_end_indx == num_down_sampled_hops_after_partition); 
        end
        c_AC_partition_start_idx = num_padded_hops_per_partition + 1 + offset_idx_for_c_AC_partition;
        c_AC_partition_end_idx = c_AC_partition_start_idx + (num_hops_down_sampled_for_this_partition - 1) * sub_sampling_factor;
        offset_idx_for_c_AC_partition = num_padded_hops_per_partition + num_hops_per_partition - c_AC_partition_end_idx;
        assert(offset_idx_for_c_AC_partition >= 0 && offset_idx_for_c_AC_partition <= sub_sampling_factor - 1);
        offset_idx_for_c_AC_partition = sub_sampling_factor - 1 - offset_idx_for_c_AC_partition;
        
        c_AC = c_AC(c_AC_partition_start_idx:sub_sampling_factor:c_AC_partition_end_idx, :, :);
        
        if db_scale
            c_AC = single(20 * log10(abs(c_AC) + 1e-10) + 200);
        else
            c_AC = single(abs(c_AC));
        end
        assert(isa(c_AC, 'single'));
        
        c_AC_ensemble(c_AC_ensemble_start_idx:c_AC_ensemble_end_indx, :, :) = c_AC;
        
        c_AC_ensemble_start_idx = c_AC_ensemble_end_indx + 1;
        
    end % partition_idx=1:num_partitions
    
    if nargout == 2
        err_db = 20 * log10(norm(x(:)) / norm(x(:) - x_recovered(:)));
    end
    
    if num_hops_original_down_sampled < num_down_sampled_hops_after_partition
        c_AC_ensemble = c_AC_ensemble(1:num_hops_original_down_sampled, :, :);
    end
end
