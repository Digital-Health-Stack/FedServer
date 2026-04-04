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
  return HTTPService.get(`/list-raw-datasets?skip=${skip}&limit=${limit}`);
};

export const getProcessedDatasets = (skip = 0, limit = 5) => {
  return HTTPService.get(`/list-datasets?skip=${skip}&limit=${limit}`);
};

/** Unified list of all datasets (raw + processed). */
export const getAllDatasets = (skip = 0, limit = 100) => {
  return HTTPService.get(`/list-all-datasets?skip=${skip}&limit=${limit}`);
};

export const viewRecentUploads = () => {
  return HTTPService.get(`/list-recent-uploads`);
};

export const getDatasetDetails = (filename: string) => {
  return HTTPService.get(`/dataset-details/${encodeURIComponent(filename)}`);
};

/** Preview first N rows of a dataset. */
export const getDatasetPreview = (filename: string, n = 5) => {
  return HTTPService.get(`/dataset-preview/${encodeURIComponent(filename)}`, {
    params: { n },
  });
};

export const getRawDatasetDetail = (filename: string) => {
  return HTTPService.get(`/raw-dataset-details/${filename}`);
};

export const createNewDataset = (data: { filename: string }) => {
  return HTTPService.post("/create-new-dataset", data);
};

export const createNewTask = (data: {
  dataset_id: number;
  task_name: string;
  output_column: string;
  metric: string;
  benchmark?: {
    [key: string]: {
      std_mean: number;
      std_dev: number;
    };
  };
}) => {
  return HTTPService.post("/create-task", data);
};

export const deleteTask = (task_id: number) => {
  return HTTPService.delete(`/delete-task/${task_id}`);
};

export const preprocessDataset = (data: any) => {
  return HTTPService.post("/preprocess-dataset", data);
};

export const listTasksFromDatasetId = (datasetId: number) => {
  return HTTPService.get(`/list-tasks-with-datasetid/${datasetId}`);
};

export const getTrainingWithBenchmark = (benchmarkId: number) => {
  return HTTPService.get(`/get-training-with-benchmarkid/${benchmarkId}`);
};

export const deleteRecentUpload = (data: {
  directory: string;
  filename: string;
}) => {
  return HTTPService.delete("/delete-recent-uploaded-file", {
    params: data,
  });
};

/** Process an already-uploaded file in storage by filename. */
export const processStoredFile = (fileName: string) => {
  return HTTPService.post("/process-stored-file", { fileName });
};

/** List files in server storage (uploads + datasets). */
export const listStorageFiles = () => {
  return HTTPService.get("/file-upload/list-files");
};

/** Delete a file from server storage. */
export const deleteStorageFile = (filename: string) => {
  return HTTPService.delete(`/file-upload/delete/${encodeURIComponent(filename)}`);
};
