function c_AC_new = pitch_shift_xian(c_AC, fbas_AC, shift_bins, Ls, threshold)
    % c_AC shape: num_frames * num_bins * num_chs

    assert(ndims(c_AC) == 2 || ndims(c_AC) == 3);
    if ndims(c_AC) == 2
        num_chs = 1;
    else 
        num_chs = size(c_AC, 3);
        assert(num_chs >= 2);
    end
    num_bins = size(c_AC, 2);
    
    assert(iscolumn(fbas_AC) && length(fbas_AC) == num_bins);
    assert(fbas_AC(1) > 0 && fbas_AC(end) < floor(Ls / 2));
    
    c_AC_new = zeros(size(c_AC));
    for ch = 1:num_chs
       c_AC_ch = c_AC(:, :, ch);
       c_AC_ch = c_AC_ch.';
       c_AC_ch = phaseUpdate(c_AC_ch, fbas_AC, shift_bins, Ls, threshold);
       %c_AC_ch = phaseUpdate_original(c_AC_ch, fbas_AC, shift_bins, Ls, 44100, threshold); 
       c_AC_ch = circshift(c_AC_ch, shift_bins, 1);
       if shift_bins > 0
          c_AC_ch(1:shift_bins, :) = 0;
       elseif shift_bins < 0
          c_AC_ch(end + shift_bins + 1:end, :) = 0;
       end
       c_AC_ch = c_AC_ch.';
       c_AC_new(:, :, ch) = c_AC_ch;
    end
end