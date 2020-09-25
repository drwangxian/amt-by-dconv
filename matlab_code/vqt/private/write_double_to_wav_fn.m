function write_double_to_wav_fn(x, file_name, sr)
    assert(ismatrix(x) && (size(x, 2) == 1 || size(x, 2) == 2));
    assert(isa(x, 'double'));
    assert(max(x, [], 'all') < 1 && min(x, [], 'all') >= -1);
    x = int16(x * 32768);
    audiowrite(file_name, x, sr, 'BitsPerSample', 16);
end