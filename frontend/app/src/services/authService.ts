import { AxiosInstance } from "axios";
import { HTTPService } from "./config";

export const login = async (credentials: {
  username: String;
  password: String;
}) => {
  return HTTPService.post("/login", credentials);
};

export const refreshAccessToken = async (
  api: AxiosInstance,
  refresh_token: String
) => {
  return api.post("/refresh-token", { refresh_token });
};

export const logout = async (api: AxiosInstance) => {
  return api.post("/logout");
};

export const getMe = async (api: AxiosInstance) => {
  return api.get("/users/me");
};
