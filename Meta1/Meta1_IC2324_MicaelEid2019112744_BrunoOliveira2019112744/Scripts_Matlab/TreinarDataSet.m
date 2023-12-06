clear all;

% path para o dir das imagens
FolderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\TrainSet';
dataFiles = dir(fullfile(FolderPath, '*.jpg'));

% arrays vazios para armazenar as imagens
imagens_em_escala_de_cinza = cell(1, numel(dataFiles));
labels = zeros(1, numel(dataFiles));

% Loop atraves das imagens, converter RBG para tons de cinza e retirar as labels pelo nome dos ficheiros 
for i = 1:numel(dataFiles)

    nome_arquivo = fullfile(FolderPath, dataFiles(i).name);
    imagem = imread(nome_arquivo);

    % converter a imagem para tons de cinza
    imagens_em_escala_de_cinza{i} = im2gray(imagem); 
    
    %definição das labels para o treino, através dos nomes dos
    %arquivos/imagens

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

% apanhe o tamanho de uma imagem em grayscale
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

% rede neural e suas configurações para o treino

% camadas e neuronios e função de treino
net = patternnet([500 500 250 100], 'traincgp');

% funções de ativação das camadas
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'tansig';

% função de ativação da camada output
%net.layers{end}.transferFcn = 'logsig';

% função de perda
net.performFcn = 'crossentropy';

% Valor de regularização
net.performParam.regularization = 0.2;

% taxa de aprendizado
net.trainParam.lr = 1;

% numero max de epochs
net.trainParam.epochs = 1000;

% treino
net = train(net, x, t);

% avaliar a rede neural
y = net(x);
perf = perform(net, t, y);
labels_preditos = vec2ind(y);

% matriz de confusion para avaliar o desempenho
plotconfusion(t, y);

save myNet.mat net 
