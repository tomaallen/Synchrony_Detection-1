

precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);

recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';

f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat))

meanF1 = @(confusionMat) mean(f1Scores(confusionMat))

threshold = 0.4

% Y_val_init = Y_val; Y_test_init = Y_test; % call only once

Y_val(Y_val_init >= threshold) = 1; Y_val(Y_val_init < threshold) =0;
Y_test(Y_test_init >= threshold) = 1; Y_test(Y_test_init < threshold) =0;
val_conf = confusionmat(Y_val_truth, Y_val)';
test_conf = confusionmat(Y_test_truth, Y_test)';
disp('val')
precision_val = precision(val_conf)
recall_val = recall(val_conf)
disp('test')
precision_tet = precision(test_conf)
recall_tet = recall(test_conf)