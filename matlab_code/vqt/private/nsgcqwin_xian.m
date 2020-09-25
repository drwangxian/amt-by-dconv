function [g,fbas, cqtbw] = nsgcqwin_xian(fbas,cqtbw,sr,Ls)
    assert(isequal(size(fbas), size(cqtbw)));
    assert(size(fbas, 2) == 1);
    assert(isscalar(sr) && mod(sr, 1) == 0);
    assert(isscalar(Ls) && mod(Ls, 1) == 0);
    assert(fbas(1) - cqtbw(1) / 2 > 0);
    assert(fbas(end) + cqtbw(end) / 2 < sr / 2);

    fbas = round(Ls * fbas / sr);
    cqtbw = round(Ls * cqtbw / sr);
    cqtbw = max(cqtbw, 4);
    assert(fbas(1) - ceil(cqtbw(1) / 2) > 0);
    assert(fbas(end) + ceil(cqtbw(end) / 2) < floor(Ls / 2));
    g = arrayfun(@(x) winfuns('hann', x), cqtbw, 'UniformOutput', 0);
end




