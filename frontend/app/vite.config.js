import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const environmentVariables = [
  "REACT_APP_SERVER_BASE_URL",
  "REACT_APP_VIEW_RECENT_UPLOADS_URL",
  "REACT_APP_RAW_DATASETS_ENDPOINT",
  "REACT_APP_RAW_DATASET_RENAME_ENDPOINT",
  "REACT_APP_DELETE_RAW_ENDPOINT",
  "REACT_APP_RAW_OVERVIEW_PATH",
  "REACT_APP_PROCESSED_DATASETS_ENDPOINT",
  "REACT_APP_PROCESSED_DATASET_RENAME_ENDPOINT",
  "REACT_APP_DELETE_PROCESSED_ENDPOINT",
  "REACT_APP_PROCESSED_OVERVIEW_PATH",
  "REACT_APP_CREATE_NEW_DATASET_URL",
  "REACT_APP_PREPROCESS_DATASET_URL",
  "REACT_APP_GET_TASKS_WITH_DATASET_ID",
  "REACT_APP_GET_TASKS_WITH_DATASET_NAME",
  "REACT_APP_GET_BENCHMARKS_WITH_TASK_ID",
  "REACT_APP_GET_BENCHMARKS_WITH_DATASET_AND_TASK_NAME",
  "REACT_APP_GET_BENCHMARKS_WITH_DATASET_AND_TASK_ID",
  "REACT_APP_GET_TRAINING_WITH_BENCHMARK_ID",
  "REACT_APP_TRAINING_DETAILS_URL",
  "REACT_APP_PRIVATE_SERVER_URL",
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
