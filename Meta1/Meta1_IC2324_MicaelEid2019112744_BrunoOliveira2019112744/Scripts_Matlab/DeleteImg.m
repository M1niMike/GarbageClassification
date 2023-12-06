% Diretorio para a pasta das imagens original
folderPath = 'C:\ISEC\1Sem\IC\TP\DataSet\Garbage Classification\plastic';

% especificar a extensão a ser removida (ex. .jpg, .png...)
extensaoParaExcluir = '.jpeg';

% lista todos os arquivos com a extensão escolhida
files = dir(fullfile(folderPath, ['*' extensaoParaExcluir]));

% Loop através de cada arquivo com a extensão
for i = 1:length(files)
    arquivoParaExcluir = fullfile(folderPath, files(i).name);
    
    % dar delete nos arquivos
    delete(arquivoParaExcluir);

    %disp(i);
end
