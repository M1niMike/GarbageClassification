clear all

%carregar a rede
load myNet.mat

% path para o dir das imagens
FolderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\TestSet';
dataFiles = dir(fullfile(FolderPath, '*.jpg'));


% arrays vazios para armazenar as imagens
imagens_em_escala_de_cinza = cell(1, numel(dataFiles));
labels = zeros(1, numel(dataFiles));

% Loop atraves das imagens, converter RBG para tons de cinza e retirar as labels pelo nome dos ficheiros
for i = 1:numel(dataFiles)
    nome_arquivo = fullfile(FolderPath, dataFiles(i).name);
    imagem = imread(nome_arquivo);
    imagens_em_escala_de_cinza{i} = im2gray(imagem); % Converte a imagem para tons de cinza
    
    if contains(dataFiles(i).name, "cardboard")
        labels(i) = 1; 
    elseif contains(dataFiles(i).name, "glass")
        labels(i) = 2; 
    elseif contains(dataFiles(i).name, "metal")
        labels(i) = 3; 
    elseif contains(dataFiles(i).name, "paper")
        labels(i) = 4; 
    elseif contains(dataFiles(i).name, "plastic")
        labels(i) = 5;
    end
end

% tamanho de uma imagem em escala de cinza
imgSize = size(imagens_em_escala_de_cinza{1});

% numero total de imagens
numImagens = numel(imagens_em_escala_de_cinza);

% matriz 2D para armazenar as imagens
x = zeros(imgSize(1) * imgSize(2), numImagens);

% matriz x com as imagens
for i = 1:numImagens
    x(:, i) = reshape(imagens_em_escala_de_cinza{i}, [], 1);
end

% labels em vetores one-hot
t = full(ind2vec(labels));

%testar a rede
y = net(x);

%plot confusion
plotconfusion(t,y);

% converter labels predict para labels originais
predicted_labels = vec2ind(y);

% calcular a matriz de confusão no conjunto de teste
confusion_matrix = confusionmat(labels, predicted_labels);


% calcular a accuracy no conjunto de teste
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));

% calcular a sensibilidade (recall) no conjunto de teste
sensitivity = confusion_matrix(1, 1) / sum(confusion_matrix(1, :));

% calcular a especificidade no conjunto de teste
specificity = confusion_matrix(2, 2) / sum(confusion_matrix(2, :));

% calcular a f-measure no conjunto de teste
precision = confusion_matrix(1, 1) / sum(confusion_matrix(:, 1));
recall = sensitivity;
f_measure = 2 * (precision * recall) / (precision + recall);

% calcular AUC no conjunto de teste
[~, ~, ~, auc] = perfcurve(labels, y(1, :), 1);

% print das metricas
fprintf('Accuracy (Test Set): %.2f%%\n', accuracy * 100);
fprintf('Sensitivity (Test Set): %.2f%%\n', sensitivity * 100);
fprintf('Specificity (Test Set): %.2f%%\n', specificity * 100);
fprintf('F-Measure (Test Set): %.2f\n', f_measure);
fprintf('AUC (Test Set): %.2f\n', auc);