function f = nsigtf_real_xian(c_DC, c_AC, c_nyq, gd, fbas, Ls, phasemode, gpu_idx)
    f = reconstruct_fft_from_coeffs_and_gd_fn(c_DC, c_AC, c_nyq, gd, fbas, Ls, phasemode, gpu_idx);
    if gpu_idx <=0
        f = real(ifft(f, [], 1)); 
    else
        try
            gpuDevice(gpu_idx);    
            f = real(gather(ifft(gpuArray(f), [], 1)));
            gpuDevice([]);
        catch ME
            gpuDevice([]);
            rethrow(ME);
        end
    end
end
