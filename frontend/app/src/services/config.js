import axios from "axios";

export const BASE_URL = process.env.REACT_APP_SERVER_BASE_URL

export const HTTPService = axios.create({
    baseURL: BASE_URL,
    timeout: 50000,
    headers: {
        "Content-Type": "application/json",
    },
});
