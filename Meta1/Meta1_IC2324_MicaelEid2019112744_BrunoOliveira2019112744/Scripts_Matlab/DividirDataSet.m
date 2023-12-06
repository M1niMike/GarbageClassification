% diretorio da pasta contendo as imagens originais
folderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\Paper_resize';

% diretorios das pastas de destino para treinamento, validação e teste
trainFolderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\TrainSet';
validationFolderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\ValidationSet';
testFolderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\TestSet';

% Defina a divisão para treinamento, validação e teste
trainRatio = 0.7;
validationRatio = 0.15;
testRatio = 0.15;

% Lista todos os arquivos de imagem na pasta
files = dir(fullfile(folderPath, '*.jpg')); % Pode ser necessário ajustar a extensão do arquivo

% Numero total de imagens
numImages = numel(files);

% Calcula o numero de imagens para cada conjunto
numTrain = round(trainRatio * numImages);
numValidation = round(validationRatio * numImages);
numTest = numImages - numTrain - numValidation;

% Gere indices aleatorios para embaralhar as imagens
indices = randperm(numImages);

% Divisão das imagens em conjuntos de treinamento, validação e teste
trainImages = files(indices(1:numTrain));
validationImages = files(indices(numTrain+1:numTrain+numValidation));
testImages = files(indices(numTrain+numValidation+1:end));


% Copie as imagens para as pastas de destino
for i = 1:numTrain
    sourceFile = fullfile(folderPath, trainImages(i).name);
    destFile = fullfile(trainFolderPath, trainImages(i).name);
    copyfile(sourceFile, destFile);
end

for i = 1:numValidation
    sourceFile = fullfile(folderPath, validationImages(i).name);
    destFile = fullfile(validationFolderPath, validationImages(i).name);
    copyfile(sourceFile, destFile);
end

for i = 1:numTest
    sourceFile = fullfile(folderPath, testImages(i).name);
    destFile = fullfile(testFolderPath, testImages(i).name);
    copyfile(sourceFile, destFile);
end