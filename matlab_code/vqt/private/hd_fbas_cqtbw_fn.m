function [fbas, cqtbw] = hd_fbas_cqtbw_fn(fmin, fmax, gamma, B, sr)
    assert(fmax < sr / 2);
    fbas = zeros(2000, 1);
    idx = 1;
    f = fmin;
    inc_factor = 2 ^ (1 / B);
    while f < fmax
        fbas(idx) = f;
        idx = idx + 1;
        f = f * inc_factor;
    end
    fbas = fbas(1:idx - 1);
    Q = 2 ^ (1 / B) - 2 ^ (-1 / B);
    cqtbw = max(gamma, fbas * Q);
    tmp_idx = find(fbas - cqtbw / 2 < 2, 1, 'last');
    if ~isempty(tmp_idx)
        cqtbw = cqtbw(tmp_idx + 1:end);
        fbas = fbas(tmp_idx + 1:end);
    end
    tmp_idx = find(fbas + cqtbw / 2 > sr / 2 - 2, 1, 'first') ;
    if ~isempty(tmp_idx)
        cqtbw = cqtbw(1:tmp_idx - 1);
        fbas = fbas(1:tmp_idx - 1);
    end
end