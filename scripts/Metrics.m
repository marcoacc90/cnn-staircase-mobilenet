clc;
clear all;

mode = 'test';
EPOCHS = 500;
folder = './../models';   % Folder with the file: mobilenet_EPOCHS_test.txt

cmd = sprintf('./../Dataset/%s/data.txt',mode);
data = readtable(cmd);
class = data{:,2};
cmd = sprintf('%s/mobilenet_%d_test.txt',folder,EPOCHS);
prediction = load(cmd);
prediction = prediction(:,1);

% 1 - obstacle
% 0 - non-obstacle
tp = 0;
tn = 0;
fp = 0;
fn = 0;
p = sum( class );
n = length(class) - p;
for i = 1 : length( class )
    if class(i) == 1 && prediction(i) == 1
        tp = tp + 1;
    elseif class(i) == 1 && prediction(i) == 0
        fn = fn + 1;
    elseif class(i) == 0 && prediction(i) == 0
        tn = tn + 1;
    else
        fp = fp + 1;
    end
end
tpr = tp/p;
fnr = fn/p;
tnr = tn/n;
fpr = fp/n;

den = sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) );
specificity = tn/(tn + fp)
sensitivity = tp/(tp + fn)
acc = (tp + tn) / (p + n)
mcc = (tp*tn - fp*fn) / den
