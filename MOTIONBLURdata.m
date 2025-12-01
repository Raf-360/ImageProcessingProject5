% =============================================================
% Author: Carlos Lopez Hernandez
% Project: Motion Blur Dataset Generator
% =============================================================
clc; clear; close all;

disp('--- Motion Blur Dataset Generator ---');

% -----------------------------
% STEP 1: Select Image
% -----------------------------
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Image Files'}, ...
                                 'Select the CLEAN X-Ray image');
if isequal(filename, 0)
    error('No image selected.');
end

I = im2double(imread(fullfile(pathname, filename)));
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% -----------------------------
% STEP 2: Select Output Folder
% -----------------------------
output_folder = uigetdir(matlabdrive, 'Select OUTPUT folder for blurred images');
if output_folder == 0
    error('No output folder selected.');
end

% -----------------------------
% STEP 3: Parameters
% -----------------------------
num_images = 50;        % how many blurred images to generate
rotation_angles = -20:5:20;   % rotate image between -20° and 20°
T = 1;                  % exposure time (fixed)

[M, N] = size(Igray);
u = (-N/2:N/2-1)/N;
v = (-M/2:M/2-1)/M;
[U,V] = meshgrid(u,v);

disp('Generating dataset...');

% -----------------------------
% STEP 4: Generate Motion Blur Images
% -----------------------------
for k = 1:num_images

    % Random motion blur direction
    a = randi([-30, 30]);    % horizontal component
    b = randi([-30, 30]);    % vertical component

    % Avoid the case a=b=0 (no blur)
    if a == 0 && b == 0
        a = 1;
    end

    % Random rotation
    angle = rotation_angles(randi(length(rotation_angles)));
    Irot = imrotate(Igray, angle, 'bilinear', 'crop');

    % Build motion blur filter H(u,v)
    H = T * (sin(pi*(U*a + V*b)) ./ (pi*(U*a + V*b))) .* exp(-1i*pi*(U*a + V*b));
    H(isnan(H)) = T;

    % Apply blur
    F = fftshift(fft2(Irot));
    G = F .* H;
    blurred = real(ifft2(ifftshift(G)));

    % Normalize to valid range
    blurred = mat2gray(blurred);

    % Save image
    filename_out = sprintf('blurred_a%d_b%d_rot%d_%03d.png', a, b, angle, k);
    imwrite(blurred, fullfile(output_folder, filename_out));

    fprintf('Saved: %s\n', filename_out);
end

disp('--- Dataset generation completed successfully ---');
