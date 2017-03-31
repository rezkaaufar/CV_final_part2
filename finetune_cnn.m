function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
run(fullfile(fileparts(mfilename('fullpath')), ...
'matconvnet-1.0-beta23', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-caltech.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.train.gpus = [];



%% update model

net = update_model();

%% TODO: Implement getCaltechIMDB function below

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCaltechIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%%
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train ;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

end

% -------------------------------------------------------------------------
function imdb = getCaltechIMDB()
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'cars', 'faces', 'motorbikes'};
splits = {'train', 'validation'};

%% TODO: Implement your loop here, to create the data structure described in the assignment
train_percent_size = 0.7;
[train_names, valid_names, test_names] = preprocess_data(train_percent_size);
data_temp = {};
labels_temp = {};
sets_temp = {};
for i=1:length(train_names)
    name = num2str(cell2mat(train_names(i)));
    im = imread(strcat('../Caltech4/ImageData/',name,'.jpg'));
    if(size(im,3) == 3)
        imrsz = imresize(im, [32 32]);
        data_temp = [data_temp,imrsz];
        if(contains(name,'airplanes') == 1)
            labels_temp = [labels_temp,1];
        elseif(contains(name,'cars') == 1)
            labels_temp = [labels_temp,2];
        elseif(contains(name,'faces') == 1)
            labels_temp = [labels_temp,3];
        elseif(contains(name,'motorbikes') == 1)
            labels_temp = [labels_temp,4];
        end
        sets_temp = [sets_temp,1];
    end
end
for i=1:length(valid_names)
    name = num2str(cell2mat(valid_names(i)));
    im = imread(strcat('../Caltech4/ImageData/',name,'.jpg'));
    if(size(im,3) == 3)
        imrsz = imresize(im, [32 32]);
        data_temp = [data_temp,imrsz];
        if(contains(name,'airplanes') == 1)
            labels_temp = [labels_temp,1];
        elseif(contains(name,'cars') == 1)
            labels_temp = [labels_temp,2];
        elseif(contains(name,'faces') == 1)
            labels_temp = [labels_temp,3];
        elseif(contains(name,'motorbikes') == 1)
            labels_temp = [labels_temp,4];
        end
        sets_temp = [sets_temp,1];
    end
end
for i=1:length(test_names)
    name = num2str(cell2mat(test_names(i)));
    im = imread(strcat('../Caltech4/ImageData/',name,'.jpg'));
    if(size(im,3) == 3)
        imrsz = imresize(im, [32 32]);
        data_temp = [data_temp,imrsz];
        if(contains(name,'airplanes') == 1)
            labels_temp = [labels_temp,1];
        elseif(contains(name,'cars') == 1)
            labels_temp = [labels_temp,2];
        elseif(contains(name,'faces') == 1)
            labels_temp = [labels_temp,3];
        elseif(contains(name,'motorbikes') == 1)
            labels_temp = [labels_temp,4];
        end
        sets_temp = [sets_temp,2];
    end
end
data = single(cat(4,data_temp{:,:,:}));
labels = single(cell2mat(labels_temp));
sets = single(cell2mat(sets_temp));

%%
% subtract mean
dataMean = mean(data(:, :, :, sets == 1), 4);
data = bsxfun(@minus, data, dataMean);

imdb.images.data = data ;
imdb.images.labels = single(labels) ;
imdb.images.set = sets;
imdb.meta.sets = splits;
imdb.meta.classes = classes;

perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);

end
