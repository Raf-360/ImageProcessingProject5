% % =============================================================
% % Full-Image DnCNN Inference Using 64x64 Tiling
% % =============================================================
% clc; clear; close all;
% 
% disp('--- DnCNN Full-Image Deblurring Inference ---');
% 
% % Load model
% [modelFile, modelPath] = uigetfile('*.mat', 'Select DnCNN_Deblur_Model.mat');
% if isequal(modelFile,0), error('No model selected.'); end
% load(fullfile(modelPath, modelFile), "net");
% disp("✔ Loaded model: " + modelFile);
% 
% % Load image
% [inputFile, inputPath] = uigetfile({'*.png;*.jpg;*.bmp;*.tif'}, ...
%                                    'Select BLURRED image');
% if isequal(inputFile,0), error('No input image selected.'); end
% 
% blurImg = im2double(imread(fullfile(inputPath, inputFile)));
% if size(blurImg,3)==3
%     blurImg = rgb2gray(blurImg);
% end
% 
% % DnCNN patch size
% ps = 64;
% 
% [M,N] = size(blurImg);
% restored = zeros(M,N);
% 
% disp('Processing full image using 64x64 tiling...');
% 
% for r = 1:ps:M-ps+1
%     for c = 1:ps:N-ps+1
% 
%         % extract patch
%         patch = blurImg(r:r+ps-1, c:c+ps-1);
%         patch4D = reshape(patch, [ps ps 1 1]);
% 
%         % predict residual
%         residual = predict(net, patch4D);
%         cleanPatch = patch - residual(:,:,1,1);
% 
%         % store
%         restored(r:r+ps-1, c:c+ps-1) = cleanPatch;
%     end
% end
% 
% % Clip range
% restored = max(min(restored,1),0);
% 
% % Display results
% figure;
% subplot(1,2,1); imshow(blurImg,[]); title('Blurred Input');
% subplot(1,2,2); imshow(restored,[]); title('Restored Output (DnCNN)');
% 
% % Save output
% [saveFile, savePath] = uiputfile('restored.png', 'Save RESTORED image as');
% if saveFile ~= 0
%     imwrite(restored, fullfile(savePath, saveFile));
%     disp("✔ Saved: " + fullfile(savePath, saveFile));
% end
% 
% disp('--- Inference Complete ---');

clc; clear; close all;

% ============================================================
% Load DnCNN Model
% ============================================================
[modelFile, modelPath] = uigetfile('*.mat', 'Select DnCNN Model');
if isequal(modelFile, 0), error("No model selected."); end
load(fullfile(modelPath, modelFile), "net");
disp("✔ Loaded model: " + modelFile);

% ============================================================
% Load Blurred + Clean Image
% ============================================================
[blurFile, blurPath] = uigetfile({'*.png;*.jpg;*.bmp'}, 'Select BLURRED image');
if isequal(blurFile,0), error("No blurred image selected."); end

[cleanFile, cleanPath] = uigetfile({'*.png;*.jpg;*.bmp'}, 'Select CLEAN reference image');
if isequal(cleanFile,0), error("No clean image selected."); end

Iblur = im2double(imread(fullfile(blurPath, blurFile)));
Iclean = im2double(imread(fullfile(cleanPath, cleanFile)));

if size(Iblur,3)==3, Iblur = rgb2gray(Iblur); end
if size(Iclean,3)==3, Iclean = rgb2gray(Iclean); end

% ============================================================
% EXTRACT MOTION PARAMETERS FROM FILENAME (a,b)
% Example: blurred_a3_b50.png
% ============================================================
% FIXED motion blur parameters (choose values)
LEN = 15;      % length of motion
THETA = 30;    % angle in degrees

PSF = fspecial("motion", LEN, THETA);



% ============================================================
% TRADITIONAL: Lucy–Richardson Deblurring
% ============================================================
NUM_ITER = 20;
I_LR = deconvlucy(Iblur, PSF, NUM_ITER);


% ============================================================
% AI: DnCNN Deblurring (64×64 Tiles)
% ============================================================
ps = 64;
[M,N] = size(Iblur);
I_AI = zeros(M,N);

for r = 1:ps:M-ps+1
    for c = 1:ps:N-ps+1
        patch = Iblur(r:r+ps-1, c:c+ps-1);
        patch4D = reshape(patch, [ps ps 1 1]);

        residual = predict(net, patch4D);
        cleanPatch = patch - residual(:,:,1,1);

        I_AI(r:r+ps-1, c:c+ps-1) = cleanPatch;
    end
end

I_AI = max(min(I_AI, 1), 0);


% ============================================================
% METRICS
% ============================================================
psnr_lr = psnr(I_LR, Iclean);
psnr_ai = psnr(I_AI, Iclean);

ssim_lr = ssim(I_LR, Iclean);
ssim_ai = ssim(I_AI, Iclean);

fprintf("\n=== METRICS ===\n");
fprintf("Lucy–Richardson PSNR: %.4f | SSIM: %.4f\n", psnr_lr, ssim_lr);
fprintf("DnCNN AI      PSNR: %.4f | SSIM: %.4f\n", psnr_ai, ssim_ai);


% ============================================================
% DISPLAY COMPARISON
% ============================================================
figure;
subplot(1,3,1); imshow(Iblur,[]); title("Blurred Input");
subplot(1,3,2); imshow(I_LR,[]); title(sprintf("Lucy–Richardson\nPSNR=%.2f SSIM=%.3f", psnr_lr, ssim_lr));
subplot(1,3,3); imshow(I_AI,[]); title(sprintf("AI (DnCNN)\nPSNR=%.2f SSIM=%.3f", psnr_ai, ssim_ai));


% ============================================================
% SAVE OUTPUTS
% ============================================================
imwrite(I_LR, "LR_Result.png");
imwrite(I_AI, "AI_Result.png");

disp("✔ Saved LR_Result.png and AI_Result.png");
