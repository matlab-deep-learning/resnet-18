function net = assembleResNet18()
% assembleResNet18   Assemble ResNet-18 network
%
% net = assembleResNet18 creates a ResNet-18 network with weights trained
% on ImageNet. You can load the same ResNet-18 network by installing the
% Deep Learning Toolbox Model for ResNet-18 Network support package from
% the Add-On Explorer and then using the resnet18 function.

%   Copyright 2019 The MathWorks, Inc.

% Download the network parameters. If these have already been downloaded,
% this step will be skipped.
%
% The files will be downloaded to a file "resnet18Params.mat", in a
% directory "ResNet18" located in the system's temporary directory.
dataDir = fullfile(tempdir, "ResNet18");
paramFile = fullfile(dataDir, "resnet18Params.mat");
downloadUrl = "http://www.mathworks.com/supportfiles/nnet/data/networks/resnet18Params.mat";

if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

if ~exist(paramFile, "file")
    disp("Downloading pretrained parameters file (43 MB).")
    disp("This may take several minutes...");
    websave(paramFile, downloadUrl);
    disp("Download finished.");
else
    disp("Skipping download, parameter file already exists.");
end

% Load the network parameters from the file resNet18Params.mat.
s = load(paramFile);
params = s.params;

% Create a layer graph with the network architecture of ResNet-18.
lgraph = resnet18Layers;

% Create a cell array containing the layer names.
layerNames = {lgraph.Layers(:).Name}';

% Loop over layers and add parameters.
for i = 1:numel(layerNames)
    name = layerNames{i};
    idx = strcmp(layerNames,name);
    layer = lgraph.Layers(idx);
    
    % Assign layer parameters.
    layerParams = params.(name);
    if ~isempty(layerParams)
        paramNames = fields(layerParams);
        for j = 1:numel(paramNames)
            layer.(paramNames{j}) = layerParams.(paramNames{j});
        end
    end
    
    % Add layer into layer graph.
    lgraph = replaceLayer(lgraph,name,layer);
end

% Assemble the network.
net = assembleNetwork(lgraph);

end