function [train_cnn_names, valid_cnn_names, test_cnn_names] = preprocess_data(train_percent_size)

%get the names of the file we want to train the model on
folder = '../Caltech4/ImageSets/';
filenames = dir(strcat(folder,'*_train.txt'));
train_cnn_names = {}; 
valid_cnn_names={};
for i=1:length(filenames)
    file=fopen(strcat(folder,filenames(i).name),'r');
    t_names = {};
    linenum=0;
    EoF = false; 
    while(~EoF)
        line= fgets(file);
        if(line== -1)
            EoF = true;
        else
            t_names = [t_names;line];
            linenum=linenum + 1;
        end
    end
    t_names= t_names(randperm(linenum));
    % retrieve percentage images to cluster descriptors
    t_size= int64(train_percent_size*linenum);
    train_cnn_names=[train_cnn_names;t_names(1:t_size)];

    % retrieve n images from each that is not used to cluster
    % descriptor
    valid_im = t_names(t_size+1:length(t_names));
    valid_cnn_names=[valid_cnn_names;valid_im];
end

% get test filenames
filenames = dir(strcat(folder,'*_test.txt'));
test_cnn_names={};
for i=1:length(filenames)
    file=fopen(strcat(folder,filenames(i).name),'r');
    t_names = {};
    linenum=0;
    EoF = false; 
    while(~EoF)
        line= fgets(file);
        if(line== -1)
            EoF = true;
        else
            t_names = [t_names;line];
            linenum=linenum + 1;
        end
    end
    test_im = t_names;
    test_cnn_names=[test_cnn_names;test_im];
end

end