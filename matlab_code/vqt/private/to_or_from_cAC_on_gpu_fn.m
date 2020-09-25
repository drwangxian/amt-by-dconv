function coeff_AC=to_or_from_cAC_on_gpu_fn(gpu_idx, fn_handle, ck)
    % to c_AC from ck that are ffts, fn_handle is ifft
    % to ck from c_AC that are iffts, fn_handle is fft

    assert(isequal(fn_handle, @fft) || isequal(fn_handle, @ifft));
    assert(ndims(ck) == 2 || ndims(ck) == 3);
    if ndims(ck) == 2
        num_chs = 1;
    else
        num_chs = size(ck, 3);
        assert(num_chs >= 2);
    end
    
    bins = size(ck, 2);
    M_AC = size(ck, 1);
    coeff_AC = complex(zeros(M_AC, bins, num_chs), zeros(M_AC, bins, num_chs));
    
    num_iters = 1;
    while true
        divisions = unique(floor(linspace(1, bins + 1, num_iters + 1)));
        assert(divisions(1) == 1 && divisions(end) == bins + 1);
        no_exception = false;
        for iter=1:num_iters
            start_bin = divisions(iter);
            end_bin = divisions(iter + 1) - 1;
            try
                gpuDevice(gpu_idx);
                tmp = gather(fn_handle(gpuArray(ck(:, start_bin:end_bin, :)),[], 1));
                if num_chs >= 2
                    tmp_size = [M_AC, end_bin - start_bin + 1, num_chs];
                else
                    tmp_size = [M_AC, end_bin - start_bin + 1];
                end
                assert(isequal(size(tmp), tmp_size));
                coeff_AC(:, start_bin:end_bin, :) = tmp;
                if iter == num_iters
                    no_exception = true; 
                end
            catch
                fprintf('division of value %d infeasible\n', num_iters);
                break;
            end
        end

        if no_exception
            gpuDevice([]);
            break;
        else
            gpuDevice([]);
            num_iters = num_iters + 1;
            if num_iters > bins
                error('ifft on gpu infeasible');
            end
        end
    end % end while
    assert(no_exception);
    gpuDevice([]);
end