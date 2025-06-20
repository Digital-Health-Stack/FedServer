import { AxiosInstance } from "axios";

export const getAllSessions = async (api, page = 1, perPage = 6) => {
  return api.get(
    `/get-all-federated-sessions?page=${page}&per_page=${perPage}`,
  );
};

export const getFederatedSession = (api: AxiosInstance, session_id) => {
  return api.get(`get-federated-session/${session_id}`);
};

export const getLogsSession = (api: AxiosInstance, session_id) => {
  return api.get(`logs/${session_id}`);
};
export const getTrainingResults = (api: AxiosInstance, session_id) => {
  return api.get(`training-result/${session_id}`);
};

export const getLeaderboardByTaskId = (api: AxiosInstance, task_id: number) => {
  return api.get(`/leaderboard/${task_id}`);
};
