fileID = fopen('train_label.txt','r');
trainlabels = fscanf(fileID, '%f');
trainlabels( find(trainlabels == -1) ) = 0;

fileID = fopen('train_data.txt', 'r');
train_data = zeros(3694,2304);
i = 1;
while ~feof(fileID)
    line = fgetl(fileID);
    feature = sscanf(line,'%f')';
    train_data(i,:) = feature;
    i = i + 1;
end

numTestImages = size(train_data, 1);
numFeatures  = size(train_data, 2);

p_male = norm(trainlabels, 1) * 1.0 / size(trainlabels, 1);
n_feature_male = ones(1 ,numFeatures);
n_feature_female = ones(1, numFeatures);
n_total_male = numFeatures;
n_total_female = numFeatures;

for i = 1 : numTestImages
  if trainlabels(i)
    for j = 1 : numFeatures
      n_feature_male(j) = n_feature_male(j) + train_data(i, j);
      n_total_male = n_total_male + train_data(i, j);
    end
  else
    for j = 1 :numFeatures
      n_feature_female(j) = n_feature_female(j) + train_data(i, j);
      n_total_female = n_total_female + train_data(i, j);
    end
  end
end

