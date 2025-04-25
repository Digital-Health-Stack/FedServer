import React, { useEffect, useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import {
  ChartBarIcon,
  ScaleIcon,
  ArrowTopRightOnSquareIcon,
} from "@heroicons/react/24/outline";
import { motion } from "framer-motion";

// Color palette for different tasks
const TASK_COLORS = [
  "bg-blue-100 text-blue-800",
  "bg-green-100 text-green-800",
  "bg-purple-100 text-purple-800",
  "bg-orange-100 text-orange-800",
  "bg-pink-100 text-pink-800",
];

const Tasks = ({ datasetId }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_GET_TASKS_WITH_DATASET_ID}/${datasetId}`
        );
        setTasks(response.data);
      } finally {
        setLoading(false);
      }
    };
    fetchTasks();
  }, [datasetId]);

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center gap-2 mb-6">
        <span className="text-2xl font-semibold">Associated Tasks</span>
      </div>

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

                  <Link
                    to={`/history/${task.task_id}`}
                    className={`text-sm flex items-center gap-1 px-4 py-2 rounded-lg ${TASK_COLORS[colorIndex]} hover:opacity-80`}
                  >
                    <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                    History
                  </Link>
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
