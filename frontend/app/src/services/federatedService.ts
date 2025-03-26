import { AxiosInstance } from "axios";

export const getAllSessions = (api: AxiosInstance) => {
  return api.get("get-all-federated-sessions");
};

export const getFederatedSession = (api: AxiosInstance, session_id) => {
  return api.get(`get-federated-session/${session_id}`);
};
