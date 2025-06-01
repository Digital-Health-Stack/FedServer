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
