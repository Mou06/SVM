imds = imageDatastore('EDATA_SEG','IncludeSubfolders',true,'LabelSource','foldernames');
%%
tbl = countEachLabel(imds)
% tbl =
% 
%   2×2 table
% 
%         Label         Count
%     ______________    _____
% 
%     NON_ROI227x227    2100 
%     ROI227x227        2110 
%%
[trainingSet, validationSet] = splitEachLabel(imds, 0.7, 'randomize');
%%
bag = bagOfFeatures(trainingSet);
% Creating Bag-Of-Features.
-------------------------
% * Image category 1: NON_ROI227x227
% * Image category 2: ROI227x227
% * Selecting feature point locations using the Grid method.
% * Extracting SURF features from the selected feature point locations.
% ** The GridStep is [8 8] and the BlockWidth is [32 64 96 128].
% 
% * Extracting features from 2947 images...done. Extracted 9241792 features.
% 
% * Keeping 80 percent of the strongest features from each category.
% 
% * Balancing the number of features across all image categories to improve clustering.
% ** Image category 1 has the least number of strongest features: 3687936.
% ** Using the strongest 3687936 features from each of the other image categories.
% 
% * Using K-Means clustering to create a 500 word visual vocabulary.
% * Number of features          : 7375872
% * Number of clusters (K)      : 500
% 
% * Initializing cluster centers...100.00%.
% * Clustering...completed 19/100 iterations (~18.80 seconds/iteration)...converged in 19 iterations.
% 
% * Finished creating Bag-Of-Features

%%
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
%Training an image category classifier for 2 categories.
% --------------------------------------------------------
% * Category 1: NON_ROI227x227
% * Category 2: ROI227x227
% 
% * Encoding features for 2947 images...done.
% 
% * Finished training the category classifier. Use evaluate to test the classifier on a test set.
%%
confMatrix = evaluate(categoryClassifier, trainingSet);

% Evaluating image category classifier for 2 categories.
% -------------------------------------------------------
% 
% * Category 1: NON_ROI227x227
% * Category 2: ROI227x227
% 
% * Evaluating 2947 images...done.
% 
% * Finished evaluating all the test sets.
% 
% * The confusion matrix for this test set is:
% 
% 
%                               PREDICTED
% KNOWN             | NON_ROI227x227   ROI227x227   
% --------------------------------------------------
% NON_ROI227x227    | 0.67             0.33         
% ROI227x227        | 0.14             0.86         
% 
% * Average Accuracy is 0.77.
%%
confMatrix = evaluate(categoryClassifier, validationSet);
% Evaluating image category classifier for 2 categories.
% -------------------------------------------------------
% 
% * Category 1: NON_ROI227x227
% * Category 2: ROI227x227
% 
% * Evaluating 1263 images...done.
% 
% * Finished evaluating all the test sets.
% 
% * The confusion matrix for this test set is:
% 
% 
%                               PREDICTED
% KNOWN             | NON_ROI227x227   ROI227x227   
% --------------------------------------------------
% NON_ROI227x227    | 0.61             0.39         
% ROI227x227        | 0.16             0.84         
% 
% * Average Accuracy is 0.72.
%%
% Compute average accuracy
mean(diag(confMatrix))
% ans =

%     0.7234
%%
img = imread(fullfile('Images','cats','cat.10.jpg'));
[labelIdx, scores] = predict(categoryClassifier, img);
% 
% % Display the string label
% categoryClassifier.Labels(labelIdx)
% 
% ans =
% 
%   1×1 cell array
% 
%     {'ROI_227x227'}
%%
%  img = imread(fullfile('EDATA','NONROI_227X227','NONROI_141010_Frame737.jpg.jpg'));
% [labelIdx, scores] = predict(categoryClassifier, img);
% categoryClassifier.Labels(labelIdx)
% 
% ans =
% 
%   1×1 cell array
% 
%     {'NONROI_227x227'}