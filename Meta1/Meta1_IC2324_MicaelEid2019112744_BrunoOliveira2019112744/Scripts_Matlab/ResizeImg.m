%Diretorio da pasta orignial
folderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage Classification\metal';

%Diretorio da pasta resize
resizeFolder = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage_Classification_Resize\Metal_resize';

% Novas dimensões
novaAltura = 28;
novaLargura = 28; 

% Listar todos os arquivos na folderPath com a extensão .jpg
files = dir(fullfile(folderPath, '*.jpg')); 

% Numero de imagens a serem redimensionadas
limiteImagens = 1300;

% Loop através de cada imagem na pasta até o limite
for i = 1:min(limiteImagens, length(files))
    
    % Leitura da imagem original
    imagemOriginal = imread(fullfile(folderPath, files(i).name));

    % Resize da imagem
    imagemRedimensionada = imresize(imagemOriginal, [novaAltura, novaLargura]);

    % Criação de um novo ficheiro/nome para cada imagem redimensionada
    % Nome deve ser alterado em hardcode (ex. metal_resize, paper_resize...)
    novoNome = fullfile(resizeFolder, ['metal_resize_' num2str(i) '.jpg']);

    % Salve a imagem redimensionada
    imwrite(imagemRedimensionada, novoNome);
    
    %disp(i);
end
