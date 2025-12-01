clc; clear; close all;

%% -------------------------------
%  Load Trained DnCNN Model
% -------------------------------
modelPath = fullfile('MATLAB Drive','DnCNN_Deblur_Model.mat');
fprintf('Loading model: %s\n', modelPath);

data = load(modelPath);
net = data.ett;   % assumes model saved as variable "net"

%% -------------------------------
%  Select Image to Deblur
% -------------------------------
[filename, pathname] = uigetfile({'*.png;*.jpg;*.jpeg', 'Images'});
if isequal(filename,0)
    error('No image selected.');
end

I = im2double(imread(fullfile(pathname, filename)));
if size(I,3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

figure; imshow(Igray,[]); title('Input Blurred Image');

%% -------------------------------
%  Run DnCNN to Deblur
% -------------------------------
% DnCNN expects single precision
I_input = single(Igray);

% Run the network
I_denoised = predict(net, I_input);

%% -------------------------------
%  Show Result
% -------------------------------
figure;
subplot(1,2,1); imshow(Igray,[]); title('Blurred Input');
subplot(1,2,2); imshow(I_denoised,[]); title('DnCNN Output (Deblurred)');

disp('Deblurring complete.');
