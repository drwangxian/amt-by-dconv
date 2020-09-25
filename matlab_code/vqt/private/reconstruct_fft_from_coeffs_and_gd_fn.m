function f = reconstruct_fft_from_coeffs_and_gd_fn(c_DC, c_AC, c_nyq, gd, fbas, Ls, phasemode, gpu_idx)
    assert(ndims(c_AC) == 2 || ndims(c_AC) == 3);
    if ndims(c_AC) == 2
        num_chs = 1;
    else
        num_chs = size(c_AC, 3);
        assert(num_chs >= 2);
    end

    assert(ismatrix(c_DC) && size(c_DC, 2) == num_chs);
    assert(ismatrix(c_nyq) && size(c_nyq, 2) == num_chs);
    assert(length(gd) == length(fbas));
    assert(iscolumn(gd) && iscolumn(fbas));
    assert(fbas(1) == 0);
    assert(fbas(end) == floor(Ls / 2) || fbas(end) == ceil(Ls / 2));
    assert(strcmp(phasemode, 'global') || strcmp(phasemode, 'local'));
    if strcmp(phasemode, 'global')
        phase_global = true;
    else
        phase_global = false;
    end

    if gpu_idx <= 0
        f = complex(zeros(Ls,num_chs), zeros(Ls,num_chs));
        num_filters = length(gd);

        for filter_idx =1:num_filters
            g_len = length(gd{filter_idx});
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            glb_to_gub = glb:gub;
            f_coords = mod(fbas(filter_idx)+ glb_to_gub, Ls) + 1;
            g_coords = mod(glb_to_gub, g_len) + 1;
            if filter_idx == 1
                ck = c_DC; 
            elseif filter_idx == num_filters
                ck = c_nyq;
            else
                ck = c_AC(:, filter_idx - 1, :);
            end

            M = size(ck, 1);
            c_coords = mod(glb_to_gub, M) + 1;

            ck = fft(ck,[],1) * M;

            if phase_global
                displace = mod(fbas(filter_idx), M);
                ck = circshift(ck, -displace, 1);
            end
            f(f_coords, :) = f(f_coords, :) + ck(c_coords, :) .* gd{filter_idx}(g_coords);
        end

        [~, gub] = g_lower_and_upper_bins_xian(Ls);
        f(Ls - gub + 1:end, :) = conj( f(gub + 1 : -1 : 2, :));
    else %use gpu
        f = complex(zeros(Ls,num_chs), zeros(Ls,num_chs));
        num_filters = length(gd);
        
        for filter_idx = [1, num_filters]
            g_len = length(gd{filter_idx});
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            glb_to_gub = glb:gub;
            f_coords = mod(fbas(filter_idx)+ glb_to_gub, Ls) + 1;
            g_coords = mod(glb_to_gub, g_len) + 1;
            if filter_idx == 1
                ck = c_DC; 
            else
                ck = c_nyq;
            end
            
            M = size(ck, 1);
            c_coords = mod(glb_to_gub, M) + 1;
            
            try
                gpuDevice(gpu_idx);
                ck = gather(fft(gpuArray(ck),[],1)) * M;
            catch ME
                gpuDevice([]);
                rethrow(ME);
            end
            
            if phase_global
                displace = mod(fbas(filter_idx), M);
                ck = circshift(ck, -displace, 1);
            end
            f(f_coords, :) = f(f_coords, :) + ck(c_coords, :) .* gd{filter_idx}(g_coords);
        end
        
        c_AC_size = size(c_AC);
        c_AC = to_or_from_cAC_on_gpu_fn(gpu_idx, @fft, c_AC);
        assert(isequal(size(c_AC), c_AC_size));
        
        M_AC = size(c_AC, 1);
        for filter_idx =2:num_filters - 1
            bin_idx = filter_idx - 1;
            g_len = length(gd{filter_idx});
            [glb, gub] = g_lower_and_upper_bins_xian(g_len);
            glb_to_gub = glb:gub;
            f_coords = mod(fbas(filter_idx)+ glb_to_gub, Ls) + 1;
            g_coords = mod(glb_to_gub, g_len) + 1;
            c_coords = mod(glb_to_gub, M_AC) + 1;
            ck = c_AC(:, bin_idx, :) * M_AC;
            if phase_global
                displace = mod(fbas(filter_idx), M_AC);
                ck = circshift(ck, -displace, 1);
            end
            f(f_coords, :) = f(f_coords, :) + ck(c_coords, :) .* gd{filter_idx}(g_coords);
        end

        [~, gub] = g_lower_and_upper_bins_xian(Ls);
        f(Ls - gub + 1:end, :) = conj( f(gub + 1 : -1 : 2, :));
    end  %end use gpu
end
