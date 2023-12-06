clear all

%carregar a rede treina
load myNet

% path para o dir das imagens
FolderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\ValidationSet';
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

% validar a rede
y = net(x);

% matriz de confusion para avaliar o desempenho
plotconfusion(t,y);