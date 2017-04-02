function [] = tSNE(features, labels)
% specify parameters
mapped_dim = 2;
initial_dim = size(features,2);
perplexity = 30;

% perform tSNE
features = full(features);
mappedX = tsne(features,[],mapped_dim,initial_dim,perplexity);
gscatter(mappedX(:,1),mappedX(:,2),labels);
end