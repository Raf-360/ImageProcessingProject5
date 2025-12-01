% =============================================================
% Author: Carlos Lopez Hernandez
% Project 5 - DnCNN Deblurring (Training Script)
% =============================================================
clc; clear; close all;

disp('--- DnCNN Deblurring Training ---');

% 1) Pick CLEAN and BLUR folders
cleanDir = uigetdir(pwd, "Select CLEAN dataset folder");
blurDir  = uigetdir(pwd, "Select BLUR dataset folder");

if isequal(cleanDir,0) || isequal(blurDir,0)
    error("Folders not selected.");
end

% 2) Image datastores
cleanDS = imageDatastore(cleanDir);
blurDS  = imageDatastore(blurDir);

% Sort so pairs align
cleanDS.Files = sort(cleanDS.Files);
blurDS.Files  = sort(blurDS.Files);

numImages = min(numel(cleanDS.Files), numel(blurDS.Files));
cleanDS.Files = cleanDS.Files(1:numImages);
blurDS.Files  = blurDS.Files(1:numImages);

disp(['Pairs found: ', num2str(numImages)]);

% 3) Training patch extraction settings
patchSize = 64;      % DnCNN uses small patches
patchesPerImage = 50; % increase if you want more data

% Extract training pairs (blur patch, clean patch)
[XTrain, YTrain] = makePatchesFromPairs(blurDS, cleanDS, patchSize, patchesPerImage);

% 4) DnCNN-like residual network
% Network predicts residual R = blurred - clean
layers = dncnnLayers(patchSize);

% 5) Training options
opts = trainingOptions("adam", ...
    "MaxEpochs", 20, ...
    "MiniBatchSize", 128, ...
    "InitialLearnRate", 1e-3, ...
    "Shuffle", "every-epoch", ...
    "Plots", "training-progress", ...
    "VerboseFrequency", 20);

% 6) Train
net = trainNetwork(XTrain, YTrain, layers, opts);

% 7) Save model
save("DnCNN_Deblur_Model.mat", "net");
disp("✅ Model saved as DnCNN_Deblur_Model.mat");

% =============================================================
% =============== Helper Functions Below ======================
% =============================================================

function [XTrain, YTrain] = makePatchesFromPairs(blurDS, cleanDS, ps, ppi)
% Returns 4-D arrays for training
    numImages = numel(blurDS.Files);
    XTrain = [];
    YTrain = [];

    for k = 1:numImages
        Ib = im2double(readimage(blurDS, k));
        Ic = im2double(readimage(cleanDS, k));

        if size(Ib,3)==3, Ib = rgb2gray(Ib); end
        if size(Ic,3)==3, Ic = rgb2gray(Ic); end

        [M,N] = size(Ib);

        for p = 1:ppi
            r = randi([1, M-ps+1]);
            c = randi([1, N-ps+1]);

            pb = Ib(r:r+ps-1, c:c+ps-1);
            pc = Ic(r:r+ps-1, c:c+ps-1);

            % residual target
            res = pb - pc;

            XTrain = cat(4, XTrain, pb);
            YTrain = cat(4, YTrain, res);
        end
    end
end


function layers = dncnnLayers(ps)
% DnCNN-style residual CNN
    numFilters = 64;
    layers = [
        imageInputLayer([ps ps 1], "Normalization","none","Name","input")

        convolution2dLayer(3, numFilters, "Padding","same","Name","conv1")
        reluLayer("Name","relu1")
    ];

    % middle layers
    for i = 2:15
        layers = [
            layers
            convolution2dLayer(3, numFilters, "Padding","same", "Name", "conv"+i)
            batchNormalizationLayer("Name","bn"+i)
            reluLayer("Name","relu"+i)
        ];
    end

    % final residual prediction
    layers = [
        layers
        convolution2dLayer(3,1,"Padding","same","Name","convFinal")
        regressionLayer("Name","output")
    ];
end
