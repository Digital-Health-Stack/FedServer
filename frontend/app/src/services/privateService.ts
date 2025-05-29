import { HTTPService } from "./config";

export const listTransferredData = (params: {
  skip: number;
  limit: number;
}) => {
  return HTTPService.get("/list-transferred-data", { params });
};

export const getTransferredDataOverview = (transferId: number) => {
  return HTTPService.get(`/transferred-data-overview/${transferId}`);
};

export const approveDataTransfer = (transferId: number) => {
  return HTTPService.post(`/approve-transferred-data/${transferId}`);
};

export const getRawDatasets = (skip = 0, limit = 5) => {
  return HTTPService.get(
    `/list-raw-datasets?skip=${skip}&limit=${limit}`
  );
};

export const getProcessedDatasets = (skip = 0, limit = 5) => {
  return HTTPService.get(`/list-datasets?skip=${skip}&limit=${limit}`);
};

export const viewRecentUploads = () => {
  return HTTPService.get(`/list-recent-uploads`);
}

export const getDatasetDetails = (datasetId: number) => {
  return HTTPService.get(`/dataset-details/${datasetId}`);
};

export const getRawDatasetDetail = (filename: string) => {
  return HTTPService.get(`/raw-dataset-details/${filename}`);
}

export const createNewDataset = (data: {
  filename: string;
}) => {
  return HTTPService.post("/create-new-dataset", data);
}

export const preprocessDataset = (data: any) => {
  return HTTPService.post("/preprocess-dataset", data);
}

export const listTasksFromDatasetId = (datasetId: number) => {
  return HTTPService.get(`/list-tasks-with-datasetid/${datasetId}`);
}

export const getTrainingWithBenchmark = (benchmarkId: number) => {
  return HTTPService.get(`/get-training-with-benchmarkid/${benchmarkId}`);
}