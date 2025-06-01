import React, { useEffect, useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import {
  ChartBarIcon,
  ScaleIcon,
  ArrowTopRightOnSquareIcon,
} from "@heroicons/react/24/outline";
import { FolderPlusIcon, TrashIcon, XCircleIcon } from "@heroicons/react/24/solid";
import { motion } from "framer-motion";
import {
  listTasksFromDatasetId,
  createNewTask,
  deleteTask,
} from "../../../../services/privateService";
import { toast } from "react-toastify";

// Color palette for different tasks
const TASK_COLORS = [
  "bg-blue-100 text-blue-800",
  "bg-green-100 text-green-800",
  "bg-purple-100 text-purple-800",
  "bg-orange-100 text-orange-800",
  "bg-pink-100 text-pink-800",
];

const METRIC_OPTIONS = [
  "Accuracy",
  "F1 Score",
  "Mean Absolute Error",
  "Mean Squared Error",
  "Precision",
  "Recall",
];

const METRIC_MAP = {
  "Accuracy": "accuracy",
  "F1 Score": "f1",
  "Mean Absolute Error": "mae",
  "Mean Squared Error": "mse",
  "Precision": "precision",
  "Recall": "recall",
};

const Tasks = ({ datasetId }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({
    dataset_id: datasetId || "",
    task_name: "",
    metric: METRIC_OPTIONS[0],
    std_mean: "",
    std_dev: "",
  });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const response = await listTasksFromDatasetId(datasetId);
        setTasks(response.data);
      } finally {
        setLoading(false);
      }
    };
    fetchTasks();
  }, [datasetId]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleToggleForm = () => {
    setShowForm((prev) => !prev);
    if (!showForm) {
      setForm({
        dataset_id: datasetId || "",
        task_name: "",
        metric: METRIC_OPTIONS[0],
        std_mean: "",
        std_dev: "",
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Validation
    if (!form.dataset_id || !form.task_name || !form.metric) {
      toast.error("Please fill all required fields.");
      return;
    }
    const stdMean = parseFloat(form.std_mean);
    const stdDev = parseFloat(form.std_dev);
    if (isNaN(stdMean) || isNaN(stdDev)) {
      toast.error("std_mean and std_dev must be valid numbers.");
      return;
    }
    setSubmitting(true);
    try {
      const backendMetric = METRIC_MAP[form.metric];
      const body = {
        dataset_id: form.dataset_id,
        task_name: form.task_name,
        metric: backendMetric,
        benchmark: {
          [backendMetric]: {
            std_mean: stdMean,
            std_dev: stdDev,
          },
        },
      };
      await createNewTask(body);
      toast.success("Task created successfully!");
      setShowForm(false);
      setForm({
        dataset_id: datasetId || "",
        task_name: "",
        metric: METRIC_OPTIONS[0],
        std_mean: "",
        std_dev: "",
      });
      setLoading(true);
      const response = await listTasksFromDatasetId(datasetId);
      setTasks(response.data);
    } catch (err) {
      toast.error("Failed to create task.");
    } finally {
      setLoading(false);
      setSubmitting(false);
    }
  };

  const handleDelete = async (task_id) => {
    if (!window.confirm("Are you sure you want to delete this task?")) return;
    setLoading(true);
    try {
      await deleteTask(task_id);
      toast.success("Task deleted successfully!");
      // Refresh the task list
      const response = await listTasksFromDatasetId(datasetId);
      setTasks(response.data);
    } catch (err) {
      toast.error("Failed to delete task.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center gap-2 mb-6 justify-between">
        <span className="text-2xl font-semibold">Associated Tasks</span>
        <button
          onClick={handleToggleForm}
          className="bg-indigo-500 text-white px-5 py-2.5 rounded-lg hover:bg-indigo-600 disabled:opacity-70 disabled:cursor-not-allowed transition-all flex items-center gap-2"
          type="button"
        >
          {showForm ? (
            <XCircleIcon className="w-4 h-4" />
          ) : (
            <FolderPlusIcon className="w-4 h-4" />
          )}
          {showForm ? "Close" : "Add Task"}
        </button>
      </div>

      {showForm && (
        <div className="mb-6 bg-gray-50 border border-indigo-500 rounded-lg p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Dataset ID
                </label>
                <input
                  type="text"
                  name="dataset_id"
                  value={form.dataset_id}
                  onChange={handleInputChange}
                  className="w-full border border-indigo-500 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Task Name
                </label>
                <input
                  type="text"
                  name="task_name"
                  value={form.task_name}
                  onChange={handleInputChange}
                  className="w-full border border-indigo-500 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                  required
                />
              </div>
              <div className="col-span-2 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Metric
                  </label>
                  <select
                    name="metric"
                    value={form.metric}
                    onChange={handleInputChange}
                    className="w-full border border-indigo-500 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    required
                  >
                    {METRIC_OPTIONS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Standard Mean
                  </label>
                  <input
                    type="number"
                    step="any"
                    name="std_mean"
                    value={form.std_mean}
                    onChange={handleInputChange}
                    className="w-full border border-indigo-500 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400 "
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">
                    Standard Deviation
                  </label>
                  <input
                    type="number"
                    step="any"
                    name="std_dev"
                    value={form.std_dev}
                    onChange={handleInputChange}
                    className="w-full border border-indigo-500 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    required
                  />
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                type="submit"
                className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-70 disabled:cursor-not-allowed transition-all"
                disabled={submitting}
              >
                {submitting ? "Submitting..." : "Create Task"}
              </button>
              <button
                type="button"
                className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-400 transition-all"
                onClick={handleToggleForm}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {loading ? (
        <div className="animate-pulse space-y-4">
          <div className="h-12 bg-gray-100 rounded-lg"></div>
          <div className="h-12 bg-gray-100 rounded-lg"></div>
        </div>
      ) : tasks.length > 0 ? (
        <div className="space-y-4">
          {tasks.map((task, index) => {
            const colorIndex = index % TASK_COLORS.length;
            const benchmark = task.benchmark[task.metric];

            return (
              <motion.div
                key={task.task_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`border p-4 rounded-lg hover:shadow-md transition-shadow ${TASK_COLORS[
                  colorIndex
                ].replace("bg", "border")}`}
              >
                <div className="flex items-center justify-between">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <div
                        className={`p-2 rounded-lg ${TASK_COLORS[colorIndex]}`}
                      >
                        <ChartBarIcon className="w-5 h-5" />
                      </div>
                      <h3 className="font-medium text-gray-900">
                        {task.task_name}
                      </h3>
                    </div>

                    <div className="flex items-center gap-4 ml-10">
                      <div className="flex items-center gap-2 text-sm">
                        <ScaleIcon className="w-4 h-4 text-gray-500" />
                        <span className="font-mono font-medium">
                          {task.metric}:
                          <span className="ml-2 text-blue-600">
                            {benchmark?.std_mean?.toFixed(2)}
                          </span>
                          ±{benchmark?.std_dev?.toFixed(2)}
                        </span>
                      </div>

                      <span className="text-sm text-gray-500">
                        Task ID: {task.task_id}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/history/${task.task_id}`}
                      className={`text-sm flex items-center gap-1 px-4 py-2 rounded-lg ${TASK_COLORS[colorIndex]} hover:opacity-80`}
                    >
                      <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                      History
                    </Link>
                    <button
                      type="button"
                      className={`text-sm flex items-center gap-1 px-4 py-2 rounded-lg bg-red-700 text-red-100 hover:opacity-80 cursor-pointer`}
                      onClick={() => handleDelete(task.task_id)}
                    >
                      <TrashIcon className="w-4 h-4" />
                      Delete Task
                    </button>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          No tasks associated with this dataset
        </div>
      )}
    </div>
  );
};

export default Tasks;
