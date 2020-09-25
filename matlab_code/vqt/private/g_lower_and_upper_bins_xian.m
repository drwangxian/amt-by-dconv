function [l, u]=g_lower_and_upper_bins_xian(N)
    if mod(N, 2) == 0
        l = -N / 2;
        u = N / 2 - 1;
    else
        l = -(N -1) / 2;
        u = (N - 1) / 2;
    end 
end

% upper_bound = floor((N - 1) / 2)
% lower_bound = -floor(N / 2)
% return C/C++ index, startting from 0