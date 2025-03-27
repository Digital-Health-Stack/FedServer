import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams, Link } from "react-router-dom";
import {
  ChartBarIcon,
  ClockIcon,
  TrashIcon,
  CogIcon,
  PlusIcon,
  ArrowTopRightOnSquareIcon,
} from "@heroicons/react/24/solid";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from "chart.js";
import "chartjs-adapter-date-fns";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

// Environment variables
const DATASET_DETAILS_URL = process.env.REACT_APP_PROCESSED_OVERVIEW_PATH;
const TASKS_ENDPOINT = process.env.REACT_APP_TASKS_ENDPOINT;
const BENCHMARKS_ENDPOINT = process.env.REACT_APP_BENCHMARKS_ENDPOINT;
const TRAINING_DETAILS_URL = process.env.REACT_APP_TRAINING_DETAILS_URL;

// LineChart Component
const LineChart = ({ benchmarks, metric }) => {
  const chartData = {
    datasets: [
      {
        label: metric,
        data: benchmarks.map((b) => ({
          x: new Date(b.created_at),
          y: b.metric_value,
        })),
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.5)",
        tension: 0.1,
        pointRadius: 5,
        pointHoverRadius: 7,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: "time",
        time: {
          unit: "day",
          tooltipFormat: "yyyy-MM-dd HH:mm",
        },
        title: {
          display: true,
          text: "Date",
        },
      },
      y: {
        title: {
          display: true,
          text: metric,
        },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          title: (context) => new Date(context[0].parsed.x).toLocaleString(),
          label: (context) => `${metric}: ${context.parsed.y.toFixed(4)}`,
        },
      },
    },
  };

  return (
    <div className="h-96 bg-white p-4 rounded-lg border border-gray-100">
      <Line data={chartData} options={options} />
    </div>
  );
};

// BenchmarkItem Component
const BenchmarkItem = ({ benchmark }) => (
  <div className="border-b border-gray-100 pb-4">
    <div className="flex justify-between items-center">
      <div className="flex items-center gap-4">
        <div className="bg-blue-50 p-2 rounded-lg">
          <ChartBarIcon className="h-6 w-6 text-blue-500" />
        </div>
        <div>
          <p className="font-semibold">{benchmark.metric_value.toFixed(4)}</p>
          <p className="text-sm text-gray-500 flex items-center gap-2">
            <ClockIcon className="h-4 w-4" />
            {new Date(benchmark.created_at).toLocaleString()}
          </p>
        </div>
      </div>
      <Link
        to={`${TRAINING_DETAILS_URL}/${benchmark.training_id}`}
        target="_blank"
        className="text-blue-500 hover:text-blue-700 flex items-center gap-2"
      >
        Training Details
        <ArrowTopRightOnSquareIcon className="h-5 w-5" />
      </Link>
    </div>
  </div>
);

// BenchmarkList Component
const BenchmarkList = ({ benchmarks }) => (
  <div className="space-y-4">
    {benchmarks.map((benchmark) => (
      <BenchmarkItem key={benchmark.benchmark_id} benchmark={benchmark} />
    ))}
  </div>
);

// TaskCard Component
const TaskCard = ({ task, isSelected, onSelect }) => {
  const [loadingBenchmarks, setLoadingBenchmarks] = useState(false);

  const handleClick = async () => {
    if (!isSelected) {
      setLoadingBenchmarks(true);
      await onSelect();
      setLoadingBenchmarks(false);
    } else {
      onSelect();
    }
  };

  return (
    <div
      className={`bg-white rounded-xl p-6 shadow-sm border transition-all cursor-pointer ${
        isSelected ? "border-blue-300" : "border-gray-100 hover:border-blue-200"
      }`}
      onClick={handleClick}
    >
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-lg font-semibold">{task.task_name}</h3>
          <p className="text-sm text-gray-500">{task.metric} metric</p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-500">
            {task.benchmarks_count} benchmarks
          </span>
          {loadingBenchmarks ? (
            <CogIcon className="h-6 w-6 animate-spin text-gray-400" />
          ) : (
            <CogIcon
              className={`h-6 w-6 text-gray-400 transition-transform ${
                isSelected ? "rotate-180" : ""
              }`}
            />
          )}
        </div>
      </div>
    </div>
  );
};

// Main Component
const DataSetOverview = () => {
  const { filename } = useParams();
  const [dataset, setDataset] = useState(null);
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch dataset details
        const datasetRes = await axios.get(
          `${DATASET_DETAILS_URL}/${filename}`
        );
        console.log(datasetRes.data);
        setDataset(datasetRes.data);

        // Fetch associated tasks
        const tasksRes = await axios.get(
          `${TASKS_ENDPOINT}?dataset_id=${datasetRes.data.dataset_id}`
        );
        setTasks(tasksRes.data);
      } catch (err) {
        setError(err.response?.data?.error || "Failed to load dataset details");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [filename]);

  const fetchBenchmarks = async (taskId) => {
    try {
      const res = await axios.get(`${BENCHMARKS_ENDPOINT}?task_id=${taskId}`);
      return res.data;
    } catch (err) {
      setError("Failed to load benchmarks");
      return [];
    }
  };

  const handleTaskSelect = async (task) => {
    if (selectedTask?.task_id === task.task_id) {
      setSelectedTask(null);
    } else {
      const benchmarks = await fetchBenchmarks(task.task_id);
      setSelectedTask({ ...task, benchmarks });
    }
  };

  if (loading)
    return (
      <div className="p-8 text-center">
        <CogIcon className="h-12 w-12 animate-spin text-gray-400 mx-auto" />
        <p className="mt-2 text-gray-600">Loading dataset details...</p>
      </div>
    );

  if (error) return <div className="p-8 text-center text-red-600">{error}</div>;
  if (!dataset)
    return (
      <div className="p-8 text-center text-gray-600">Dataset not found</div>
    );

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-8">
      {/* Dataset Header */}
      <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          {dataset.filename}
        </h1>
        <div className="grid grid-cols-3 gap-4 text-sm text-gray-600">
          <div>
            <p>
              <span className="font-semibold">Created:</span>{" "}
              {new Date(dataset.created_at).toLocaleDateString()}
            </p>
            <p>
              <span className="font-semibold">Columns:</span>{" "}
              {Object.keys(dataset.datastats?.column_stats || {}).length}
            </p>
          </div>
          <div>
            <p>
              <span className="font-semibold">Total Rows:</span>{" "}
              {dataset.datastats?.total_rows}
            </p>
            <p>
              <span className="font-semibold">Tasks:</span> {tasks.length}
            </p>
          </div>
        </div>
      </div>

      {/* Tasks Section */}
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold text-gray-800">
            Associated Tasks
          </h2>
          <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 flex items-center gap-2">
            <PlusIcon className="h-5 w-5" />
            Create New Task
          </button>
        </div>

        {tasks.map((task) => (
          <TaskCard
            key={task.task_id}
            task={task}
            isSelected={selectedTask?.task_id === task.task_id}
            onSelect={() => handleTaskSelect(task)}
          />
        ))}
      </div>

      {/* Selected Task Details */}
      {selectedTask && (
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="text-xl font-semibold">
              {selectedTask.task_name} Metrics
            </h3>
            <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
              {selectedTask.metric}
            </span>
          </div>

          <LineChart
            benchmarks={selectedTask.benchmarks}
            metric={selectedTask.metric}
          />

          <BenchmarkList benchmarks={selectedTask.benchmarks} />
        </div>
      )}
    </div>
  );
};

export default DataSetOverview;

// once re-designed, i'll seperate the components into their own files
