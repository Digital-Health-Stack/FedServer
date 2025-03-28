import React, { useEffect, useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import { ChartBarIcon } from "@heroicons/react/24/outline";

const TASKS_URL = process.env.REACT_APP_GET_TASKS_WITH_DATASET_ID;

const Tasks = ({ datasetId }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTasks = async () => {
      const response = await axios.get(`${TASKS_URL}/${datasetId}`);
      setTasks(response.data);
      setLoading(false);
    };
    fetchTasks();
  }, [datasetId]);

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Associated Tasks</h2>

      {loading ? (
        <p>Loading tasks...</p>
      ) : tasks.length > 0 ? (
        <div className="space-y-3">
          {tasks.map((task) => (
            <div
              key={task.task_id}
              className="border p-3 rounded-lg hover:bg-gray-50"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ChartBarIcon className="w-5 h-5 text-blue-500" />
                  <span className="font-medium">{task.task_name}</span>
                  <span className="text-sm text-gray-500">
                    ID: {task.task_id}
                  </span>
                </div>
                <Link
                  to={`/tasks/${task.task_id}/benchmarks`}
                  target="_blank"
                  className="text-blue-500 hover:text-blue-700 text-sm flex items-center gap-1"
                >
                  View Benchmarks →
                </Link>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500">No tasks found for this dataset</p>
      )}
    </div>
  );
};

export default Tasks;
