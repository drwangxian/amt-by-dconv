function [coeff_DC, coeff_AC, coeff_nyq] = nsgtf_real_xian(f, g, fbas, M_DC, M_AC, M_nyq, phasemode, gpu_idx)
    assert(size(f, 1) > size(f, 2));
    assert(iscell(g) && iscolumn(g));
    assert(iscolumn(fbas));
    assert(length(g) == length(fbas));
    
    assert(strcmp(phasemode, 'global') || strcmp(phasemode, 'local'));
    if strcmp(phasemode, 'global')
        phase_global = true;
    else
        phase_global = false;
    end
    
    [num_samples, num_chs] = size(f);
    num_filters = length(g);
    bins = num_filters - 2;
    assert(fbas(1) == 0);
    assert(fbas(end) == floor(num_samples / 2) || fbas(end) == ceil(num_samples / 2));
    
    if gpu_idx <= 0
        coeff_AC = complex(zeros(M_AC, bins, num_chs), zeros(M_AC, bins, num_chs));

        if isreal(f)
            f = fft(f, [], 1);
        end
        
        for bin_idx = 1:bins
            filter_idx = bin_idx + 1;
            ck = complex(zeros(M_AC, num_chs), zeros(M_AC, num_chs));
            g_len = length(g{filter_idx});
            assert(M_AC >= g_len);
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            assert(gub - glb + 1 == g_len);
            glb_to_gub = glb:gub;
            g_coords = mod(glb_to_gub, g_len) + 1;
            f_coords = mod(fbas(filter_idx) + glb_to_gub , num_samples) + 1;
            ck_coords = mod(glb_to_gub, M_AC) + 1;
            ck(ck_coords, :) = f(f_coords, :) .* g{filter_idx}(g_coords);
            if phase_global
                displace = mod(fbas(filter_idx), M_AC);
                ck = circshift(ck, displace, 1);
            end
            coeff_AC(:, bin_idx, :) = ifft(ck, [], 1);
        end

        for filter_idx = [1, num_filters]
            if filter_idx == 1
                m = M_DC;
            else
                m = M_nyq;
            end
            ck = complex(zeros(m, num_chs), zeros(m, num_chs));
            g_len = length(g{filter_idx});
            assert(m >= g_len);
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            glb_to_gub = glb:gub;
            g_coords = mod(glb_to_gub, g_len) + 1;
            c_coords = mod(glb_to_gub, m) + 1;
            f_coords = mod(fbas(filter_idx) + glb_to_gub, num_samples) + 1;
            ck(c_coords, :) = f(f_coords, :) .* g{filter_idx}(g_coords);
            if phase_global
               displace = mod(fbas(filter_idx), m);
               ck = circshift(ck, displace, 1);
            end

            ck = ifft(ck, [], 1);
            if filter_idx == 1
                coeff_DC = ck;
            else
                coeff_nyq = ck;
            end
        end
    else % use gpu
        coeff_AC = complex(zeros(M_AC, bins, num_chs), zeros(M_AC, bins, num_chs));

        if isreal(f)
            try
                gpuDevice(gpu_idx);
                f = gather(fft(gpuArray(f), [], 1));
            catch ME
                gpuDevice([]);
                rethrow(ME);
            end
        end

        ck_tensor = complex(zeros([M_AC, bins, num_chs]), zeros([M_AC, bins, num_chs]));
        for bin_idx = 1:bins
            filter_idx = bin_idx + 1;
            ck = complex(zeros(M_AC, num_chs), zeros(M_AC, num_chs));
            g_len = length(g{filter_idx});
            assert(M_AC >= g_len);
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            assert(gub - glb + 1 == g_len);
            glb_to_gub = glb:gub;
            g_coords = mod(glb_to_gub, g_len) + 1;
            f_coords = mod(fbas(filter_idx) + glb_to_gub , num_samples) + 1;
            ck_coords = mod(glb_to_gub, M_AC) + 1;
            ck(ck_coords, :) = f(f_coords, :) .* g{filter_idx}(g_coords);
            if phase_global
                displace = mod(fbas(filter_idx), M_AC);
                ck = circshift(ck, displace, 1);
            end
            ck_tensor(:, bin_idx, :) = ck;
        end
        coeff_AC = to_or_from_cAC_on_gpu_fn(gpu_idx, @ifft, ck_tensor);
        
        for filter_idx = [1, num_filters]
            if filter_idx == 1
                m = M_DC;
            else
                m = M_nyq;
            end
            ck = complex(zeros(m, num_chs), zeros(m, num_chs));
            g_len = length(g{filter_idx});
            assert(m >= g_len);
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            glb_to_gub = glb:gub;
            g_coords = mod(glb_to_gub, g_len) + 1;
            c_coords = mod(glb_to_gub, m) + 1;
            f_coords = mod(fbas(filter_idx) + glb_to_gub, num_samples) + 1;
            ck(c_coords, :) = f(f_coords, :) .* g{filter_idx}(g_coords);
            if phase_global
               displace = mod(fbas(filter_idx), m);
               ck = circshift(ck, displace, 1);
            end
            
            try
                gpuDevice(gpu_idx);
                ck = gather(ifft(gpuArray(ck), [], 1));
            catch ME
                gpuDevice([]);
                rethrow(ME);
            end
            if filter_idx == 1
                coeff_DC = ck;
                assert(isequal(size(coeff_DC), [M_DC, num_chs]));
            else
                coeff_nyq = ck;
                assert(isequal(size(coeff_nyq), [M_nyq, num_chs]));
            end
        end
        gpuDevice([]);
    end
end
    
    
   
    
    



























