fileID = fopen('test_label.txt','r');
test_labels = fscanf(fileID, '%f');
test_labels( find(test_labels == -1) ) = 0;

fileID = fopen('test_data.txt', 'r');
test_data = zeros(55,2304);
i = 1;
while ~feof(fileID)
    line = fgetl(fileID);
    feature = sscanf(line,'%f')';
    test_data(i,:) = feature;
    i = i + 1;
end

numTestImage = size(test_data, 1);
numFeatures  = size(test_data, 2);
output = zeros(numTestImage, 1);

for i=1:numTestImage
  p_male = 0.0;
  p_female = 0.0;
  total_features = 0;
  for j=1:numFeatures
    p_male = p_male + test_data(i,j) * log(n_feature_male(j));
    p_female = p_female + test_data(i,j) * log(n_feature_female(j));
    total_features = total_features + test_data(i, j);
  end
  p = p_male - total_features*log(n_total_male) - ...
    (p_female - total_features*log(n_total_female));
  if p > 0
    output(i, 1) = 0;
  else
    output(i, 1) = 1;
  end
end
% Compute the error on the test set
error=0;
for i=1:numTestImage
  if (test_labels(i) ~= output(i))
    error=error+1;
  end
end
%Print out the classification error on the test set
fprintf('%f\n', error/numTestImage)

