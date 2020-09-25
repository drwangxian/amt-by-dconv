function gd = nsdual_xian(g, fbas, Ms, Ls)
    assert(length(g) == length(fbas) && length(g) == length(Ms));
    assert(iscolumn(fbas) && iscolumn(Ms));
    assert(iscolumn(g));
    assert(fbas(1) == 0);
    assert(fbas(end) == floor(Ls / 2) || fbas(end) == ceil(Ls / 2));

    num_filters = length(fbas);
    diagonal=zeros(Ls,1);
    for filter_idx = 1:num_filters
        g_len = length(g{filter_idx});
        [glb, gub] = g_lower_and_upper_bins_xian(g_len);
        glb_to_gub = glb:gub;
        g_coords = mod(glb_to_gub, g_len) + 1;
        d_coords = mod(fbas(filter_idx) + glb_to_gub, Ls) + 1;
        incre = Ms(filter_idx) * g{filter_idx}(g_coords) .^ 2;
        diagonal(d_coords) = diagonal(d_coords) + incre;
        if filter_idx ~= 1 && filter_idx ~= num_filters
           d_coords = mod((Ls - fbas(filter_idx)) + glb_to_gub, Ls) + 1;
           diagonal(d_coords) = diagonal(d_coords) + incre;
        end

    end

    gd = cell(size(g));
    for filter_idx=1:num_filters
        gtmp = zeros(size(g{filter_idx}));
        g_len = length(g{filter_idx});
        [glb, gub] = g_lower_and_upper_bins_xian(g_len);
        glb_to_gub = glb:gub;
        g_coords = mod(glb_to_gub, g_len) + 1;
        d_coords = mod(fbas(filter_idx) + glb_to_gub, Ls) + 1;
        gtmp(g_coords) = g{filter_idx}(g_coords) ./ diagonal(d_coords);
        gd{filter_idx} = gtmp;
    end
end
