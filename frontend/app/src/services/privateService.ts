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
