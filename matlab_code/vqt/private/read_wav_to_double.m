function x = read_wav_to_double(wav_file)
    x = audioread(wav_file, 'native');
    assert(isa(x, 'int16'));
    assert(size(x, 2) == 2);
    x = double(x) / 32768;
end