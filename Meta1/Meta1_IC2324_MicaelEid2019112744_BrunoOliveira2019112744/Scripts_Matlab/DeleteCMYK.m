
% diretorio da pasta contendo as imagens
folderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage Classification\plastic';

% listar todos os arquivos de imagem na pasta de acordo com a extensão
% solicitada
files = dir(fullfile(folderPath, '*.jpg'));

% loop através de cada imagem na pasta
for i = 1:length(files)

    % Carregue informações da imagem usando imfinfo
    info = imfinfo(fullfile(folderPath, files(i).name));

    % verificar o espaço de cores da imagem
    if strcmp(info.ColorType, 'CMYK')

        % se for cmyk, o script apaga o ficheiro
        arquivoParaExcluir = fullfile(folderPath, files(i).name);
        delete(arquivoParaExcluir);
        disp(['Arquivo CMYK excluído: ' arquivoParaExcluir]);
    end
end
