// Written by Maor Mizrachi
// Reading HDF5 files in matlab, with links, without groups

function dataStruct = readHDF5Data(filePath)
    dataStruct = readHDF5FromFile(filePath, struct());
end

function dataStruct = readHDF5FromFile(filePath, dataStruct)
    fileID = H5F.open(filePath, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    info = h5info(filePath);
    H5F.close(fileID);

    for(i=1:length(info.Links))
        path = info.Links(i).Value{1};
        dataStruct = readHDF5FromFile(path, dataStruct);
    end

    for(i=1:length(info.Datasets))
        datasetName = info.Datasets(i).Name;
        cleanDatasetName = strrep(datasetName, ' ', '');
        data = h5read(filePath, "/" + datasetName);
        if isfield(dataStruct, cleanDatasetName)
            dataStruct.(cleanDatasetName) = vertcat(dataStruct.(cleanDatasetName), data);
        else
            dataStruct.(cleanDatasetName) = data;
        end
    end
end
