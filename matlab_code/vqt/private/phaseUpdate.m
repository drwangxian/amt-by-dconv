function c_AC_new = phaseUpdate(c_AC, fbas_AC, shift_bins, Ls, threshold)

%PHASEUPDATE Modify phases of Constant-Q/Variable-Q representation to 
%            retain phase coherence after coefficient shift
%            (transposition). For readability reasons a fully
%            rasterized representation is needed. However, the same 
%            procedure can be implemented based on a piecewise rasterized
%            representation.
%
%   Usage:  Y = phaseUpdate(c, fbas, shiftBins, xlen, fs, threshold)
%
%   Input parameters:
%         c         : transform coefficients
%         fbas      : center frequencies of filters
%         shiftBins : pitch-shifting factor in CQT bins
%         xlen      : length of input signal
%         fs        : sampling rate
%         threshold : mininum amplitude for peak-picking
%
%   Output parameters: 
%         Y         : modified transform coefficients
%
%   See also:  cqt, icqt
%
%   References:
%     C. Schörkhuber, A. Klapuri, and A. Sontacchi. Audio Pitch Shifting 
%     Using the Constant-Q Transform.
%     C. Schörkhuber, A. Klapuri, and A. Sontacchi. Pitch Shifting of Audio 
%     Signals Using the Constant-Q Trasnform.
%     C. Schörkhuber, A. Klapuri, N. Holighaus, and M. Dörfler. A Matlab 
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
%     Transforms.
%
%
% Copyright (C) 2011 Christian Schörkhuber.
% 
% This work is licensed under the Creative Commons 
% Attribution-NonCommercial-ShareAlike 3.0 Unported 
% License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc-sa/3.0/ 
% or send a letter to 
% Creative Commons, 444 Castro Street, Suite 900, 
% Mountain View, California, 94041, USA.

% Authors: Christian Schörkhuber
% Date: 10.12.2011

%c_AC shape: num_bins * num_frames
[num_bins, num_frames] = size(c_AC);

assert(ismatrix(c_AC) && num_bins == length(fbas_AC) && iscolumn(fbas_AC));
assert(fbas_AC(1) > 0 && fbas_AC(end) < floor(Ls / 2));

assert(mod(Ls, num_frames) == 0);
hop_size = 2 * pi / num_frames;

mag = abs(c_AC);



c_AC_new = zeros(size(c_AC));
c_AC_new(:,1) = c_AC(:,1);
acc_rot_angles=zeros(num_bins, 1);


for frame_idx = 2:size(c_AC,2)
   
    %peak picking
    af = mag(:,frame_idx);
    dl = af(2:end-1) - af(1:end-2) > 0;
    dr = af(2:end-1) - af(3:end) > 0;
    th = af(2:end-1) > threshold;
    peaks = find(dl & dr & th) + 1;
    if shift_bins > 0
        tmp_idx = find(peaks >= num_bins - shift_bins, 1);
        if ~isempty(tmp_idx)
            peaks = peaks(1:tmp_idx - 1);
        end
    elseif shift_bins < 0
        tmp_idx = find(peaks <= -shift_bins, 1, 'last');
        if ~isempty(tmp_idx)
            peaks = peaks(tmp_idx + 1:end);
        end
    end
    
    if ~isempty(peaks)
        
       %find regions of influence around peaks
       regions = round(0.5 * (peaks(1:end - 1) + peaks(2:end)));  
       regions = [0; regions; num_bins];
       
       %update phases 
       %(one rotation value for each region = vertical phase locking)
       shift_freq = fbas_AC(peaks + shift_bins) - fbas_AC(peaks);
       rot_angles_for_peaks = acc_rot_angles(peaks) + hop_size * shift_freq;  
       for peak_idx=1:length(peaks)
            acc_rot_angles(regions(peak_idx) + 1:regions(peak_idx + 1)) = rot_angles_for_peaks(peak_idx);
       end
       acc_rot_angles = mod(acc_rot_angles, 2 * pi);
       c_AC_new(:, frame_idx) = c_AC(:,frame_idx) .* exp(1i * acc_rot_angles);
    else
        c_AC_new(:, frame_idx) = c_AC(:, frame_idx); %if no peaks are found
    end
end