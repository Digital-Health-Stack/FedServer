import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const environmentVariables = [
  "REACT_APP_REGISTER_CLIENT_URL",
  "REACT_APP_SERVER_BASE_URL",
  "REACT_APP_GET_ALL_TRAININGS_URL",
  "REACT_APP_GET_TRAINING_RESULT_WITH_SESSION_ID_URL",
  "REACT_APP_VIEW_RECENT_UPLOADS_URL",
  "REACT_APP_VIEW_ALL_DATASETS_URL",
  "REACT_APP_DELETE_DATAFILE_URL",
  "REACT_APP_LIST_ALL_DATASETS_URL",
  "REACT_APP_DATASET_DETAILS_URL",
  "REACT_APP_PREPROCESS_DATASET_URL",
  "REACT_APP_CREATE_NEW_DATASET_URL",
];

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const processEnv = {};
  environmentVariables.forEach((key) => (processEnv[key] = env[key]));

  return {
    define: {
      "process.env": processEnv,
    },
    plugins: [react(), tailwindcss()],
    server: {
      host: "0.0.0.0", // Allows external access from Docker
      port: 5173, // Change to the new port
    },
  };
});
