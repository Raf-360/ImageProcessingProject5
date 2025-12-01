clc; clear; close all;

% ===========================================
% LUCY–RICHARDSON DEBLURRING FOR DATASET
% ===========================================

inputFolder = "MOTIONBLUR_ROTATIONDATASET";    % <-- your blurred images
outputFolder = "MOTION_DEBLURRED";      % <-- output save folder
if ~exist(outputFolder, "dir")
    mkdir(outputFolder);
end

% -------------------------------------------
% Loop Through All Blurred Images
% -------------------------------------------
files = dir(fullfile(inputFolder, "*.png"));  % change to JPG if needed

for k = 1:length(files)

    % Load image
    imgPath = fullfile(files(k).folder, files(k).name);
    I = im2double(imread(imgPath));

    if size(I,3) == 3
        I = rgb2gray(I);
    end

    % ---------------------------------------
    % EXTRACT MOTION PARAMETERS FROM FILENAME
    % Example filename: blurred_a3_b50_rot15.png
    % ---------------------------------------

    name = files(k).name;
    tokens = regexp(name, 'a(\-?\d+)_b(\-?\d+)', 'tokens');

    if isempty(tokens)
        warning("Filename missing a/b parameters: %s", name);
        continue;
    end
    
    a = str2double(tokens{1}{1});
    b = str2double(tokens{1}{2});

    % ---------------------------------------
    % BUILD MOTION BLUR PSF FROM a AND b
    % ---------------------------------------
    LEN = sqrt(a^2 + b^2);  % magnitude of motion
    THETA = atan2(b, a) * 180/pi;  % angle in degrees

    if LEN < 1
        LEN = 1; 
    end

    PSF = fspecial("motion", LEN, THETA);

    % ---------------------------------------
    % APPLY LUCY–RICHARDSON DECONVOLUTION
    % ---------------------------------------
    NUM_ITER = 20;  % You can tune this (10–30 typical)
    restored = deconvlucy(I, PSF, NUM_ITER);

    % ---------------------------------------
    % SAVE OUTPUT
    % ---------------------------------------
    outName = sprintf("deblurred_%s", name);
    imwrite(restored, fullfile(outputFolder, outName));

    fprintf("Processed: %s --> %s\n", name, outName);

end

disp("=== Lucy–Richardson Deblurring Completed ===");
