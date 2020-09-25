function [g, fbas] = nsgcqwin_with_dc_and_nyquist_fn(fbas, cqtbw, sr, Ls)
    assert(isequal(size(fbas), size(cqtbw)));
    assert(iscolumn(fbas));
    assert(isscalar(sr) && mod(sr, 1) == 0);
    assert(isscalar(Ls) && mod(Ls, 1) == 0);
    assert(fbas(1) - cqtbw(1) / 2 > 0);
    assert(fbas(end) + cqtbw(end) / 2 < sr / 2);
    
    fbas = round(Ls * fbas / sr);
    assert(length(fbas) == length(unique(fbas)));
    cqtbw = round(Ls * cqtbw / sr);
    cqtbw = max(cqtbw, 4);
    assert(fbas(1) - ceil(cqtbw(1) / 2) > 0);
    assert(fbas(end) + ceil(cqtbw(end) / 2) < floor(Ls / 2));
    cqtbw = [2 * fbas(1); cqtbw; Ls - 2 * fbas(end)];
    fbas = [0; fbas; floor(Ls / 2)];
  
    g = arrayfun(@(x) winfuns('hann', x), cqtbw, 'UniformOutput', 0);
    
    % Setup Tukey window for 0- and Nyquist-frequency
    if cqtbw(1) > cqtbw(2)
       [~, ub1] = g_lower_and_upper_bins_xian(cqtbw(1));
       [~, ub2] = g_lower_and_upper_bins_xian(cqtbw(2));
       start_idx = ub1 - ub2;
       end_idx = start_idx + cqtbw(2) - 1;
       start_idx = start_idx + 1;
       end_idx = end_idx + 1;
       g{1} = ones(cqtbw(1), 1);
       g{1}(start_idx:end_idx) = g{2};
       g{1} = g{1} / sqrt(cqtbw(1));
    end
    
    if cqtbw(end) > cqtbw(end - 1)
        [~, ub1] = g_lower_and_upper_bins_xian(cqtbw(end));
        [~, ub2] = g_lower_and_upper_bins_xian(cqtbw(end - 1));
        start_idx = ub1 - ub2;
        end_idx = start_idx + cqtbw(end - 1) - 1;
        start_idx = start_idx + 1;
        end_idx = end_idx + 1;
        g{end} = ones(cqtbw(end), 1);
        g{end}(start_idx:end_idx) = g{end - 1}; 
        g{end} = g{end} / sqrt(cqtbw(end));
    end
end