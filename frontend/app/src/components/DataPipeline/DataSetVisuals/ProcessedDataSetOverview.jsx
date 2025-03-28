// import React, { useEffect, useState } from "react";
// import axios from "axios";
// import { useParams, Link } from "react-router-dom";
// import {
//   ChartBarIcon,
//   ClockIcon,
//   CogIcon,
//   PlusIcon,
//   ArrowTopRightOnSquareIcon,
//   ChevronLeftIcon,
//   ChevronRightIcon,
// } from "@heroicons/react/24/solid";
// import { Line } from "react-chartjs-2";
// import {
//   Chart as ChartJS,
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   Title,
//   Tooltip,
//   Legend,
//   TimeScale,
// } from "chart.js";
// import "chartjs-adapter-date-fns";
// import SummaryStats from "./DatasetDetails/SummaryStats";
// import ColumnDetails from "./DatasetDetails/ColumnDetails";

// ChartJS.register(
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   Title,
//   Tooltip,
//   Legend,
//   TimeScale
// );

// const DATASET_DETAILS_URL = process.env.REACT_APP_PROCESSED_OVERVIEW_PATH;
// const TASKS_BY_DATASET_ID = process.env.REACT_APP_GET_TASKS_WITH_DATASET_ID;
// const BENCHMARKS_BY_TASK_ID = process.env.REACT_APP_GET_BENCHMARKS_WITH_TASK_ID;
// const TRAINING_DETAILS_URL =
//   process.env.REACT_APP_GET_TRAINING_WITH_BENCHMARK_ID;

// const ColumnCarousel = ({ columns }) => {
//   const [activeIndex, setActiveIndex] = useState(0);

//   if (!columns || columns.length === 0) return null;

//   return (
//     <div className="bg-white rounded-lg border border-gray-100 relative">
//       <div className="flex justify-between items-center mb-4">
//         <h3 className="text-lg font-semibold">{columns[activeIndex].name}</h3>
//         <div className="flex gap-2">
//           <button
//             onClick={() =>
//               setActiveIndex((prev) =>
//                 prev > 0 ? prev - 1 : columns.length - 1
//               )
//             }
//             className="p-1 hover:bg-gray-100 rounded"
//           >
//             <ChevronLeftIcon className="h-5 w-5 text-gray-600" />
//           </button>
//           <span className="text-sm text-gray-600">
//             {activeIndex + 1} / {columns.length}
//           </span>
//           <button
//             onClick={() =>
//               setActiveIndex((prev) =>
//                 prev < columns.length - 1 ? prev + 1 : 0
//               )
//             }
//             className="p-1 hover:bg-gray-100 rounded"
//           >
//             <ChevronRightIcon className="h-5 w-5 text-gray-600" />
//           </button>
//         </div>
//       </div>
//       <ColumnDetails columnStats={[columns[activeIndex]]} />
//     </div>
//   );
// };

// const LineChart = ({ benchmarks, metric }) => {
//   if (!benchmarks || benchmarks.length === 0) {
//     return (
//       <div className="h-96 bg-white p-4 rounded-lg border border-gray-100 flex items-center justify-center">
//         <p className="text-gray-500">No benchmark data available</p>
//       </div>
//     );
//   }

//   const chartData = {
//     datasets: [
//       {
//         label: metric,
//         data: benchmarks.map((b) => ({
//           x: new Date(b.created_at),
//           y: b.metric_value,
//         })),
//         borderColor: "#3b82f6",
//         backgroundColor: "rgba(59, 130, 246, 0.5)",
//         tension: 0.1,
//         pointRadius: 5,
//         pointHoverRadius: 7,
//       },
//     ],
//   };

//   const options = {
//     responsive: true,
//     maintainAspectRatio: false,
//     scales: {
//       x: {
//         type: "time",
//         time: {
//           unit: "day",
//           tooltipFormat: "yyyy-MM-dd HH:mm",
//         },
//         title: { display: true, text: "Date" },
//       },
//       y: { title: { display: true, text: metric } },
//     },
//     plugins: {
//       tooltip: {
//         callbacks: {
//           title: (context) => new Date(context[0].parsed.x).toLocaleString(),
//           label: (context) => `${metric}: ${context.parsed.y.toFixed(4)}`,
//         },
//       },
//     },
//   };

//   return (
//     <div className="h-96 bg-white p-4 rounded-lg border border-gray-100">
//       <Line data={chartData} options={options} />
//     </div>
//   );
// };

// const BenchmarkItem = ({ benchmark }) => (
//   <div className="border-b border-gray-100 pb-4">
//     <div className="flex justify-between items-center">
//       <div className="flex items-center gap-4">
//         <div className="bg-blue-50 p-2 rounded-lg">
//           <ChartBarIcon className="h-6 w-6 text-blue-500" />
//         </div>
//         <div>
//           <p className="font-semibold">{benchmark.metric_value.toFixed(4)}</p>
//           <p className="text-sm text-gray-500 flex items-center gap-2">
//             <ClockIcon className="h-4 w-4" />
//             {new Date(benchmark.created_at).toLocaleString()}
//           </p>
//         </div>
//       </div>
//       <Link
//         to={`${TRAINING_DETAILS_URL}/${benchmark.benchmark_id}`}
//         target="_blank"
//         className="text-blue-500 hover:text-blue-700 flex items-center gap-2"
//       >
//         Training Details
//         <ArrowTopRightOnSquareIcon className="h-5 w-5" />
//       </Link>
//     </div>
//   </div>
// );

// const BenchmarkList = ({ benchmarks }) => (
//   <div className="space-y-4">
//     {benchmarks?.length > 0 ? (
//       benchmarks.map((benchmark) => (
//         <BenchmarkItem key={benchmark.benchmark_id} benchmark={benchmark} />
//       ))
//     ) : (
//       <div className="text-center text-gray-500 py-4">
//         No benchmarks available
//       </div>
//     )}
//   </div>
// );

// const TaskCard = ({ task, isSelected, onSelect }) => {
//   const [loading, setLoading] = useState(false);

//   const handleClick = async () => {
//     if (!isSelected) {
//       setLoading(true);
//       await onSelect();
//       setLoading(false);
//     } else {
//       onSelect();
//     }
//   };

//   return (
//     <div
//       className={`bg-white rounded-xl p-6 shadow-sm border transition-all cursor-pointer ${
//         isSelected ? "border-blue-300" : "border-gray-100 hover:border-blue-200"
//       }`}
//       onClick={handleClick}
//     >
//       <div className="flex justify-between items-center">
//         <div>
//           <h3 className="text-lg font-semibold">{task.task_name}</h3>
//           <p className="text-sm text-gray-500">{task.metric} metric</p>
//         </div>
//         <div className="flex items-center gap-4">
//           <span className="text-sm text-gray-500">
//             {task.benchmarks_count} benchmarks
//           </span>
//           {loading ? (
//             <CogIcon className="h-6 w-6 animate-spin text-gray-400" />
//           ) : (
//             <CogIcon
//               className={`h-6 w-6 text-gray-400 transition-transform ${
//                 isSelected ? "rotate-180" : ""
//               }`}
//             />
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// const ProcessedDataSetOverview = () => {
//   const { filename } = useParams();
//   const [dataset, setDataset] = useState(null);
//   const [tasks, setTasks] = useState([]);
//   const [selectedTask, setSelectedTask] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         const datasetRes = await axios.get(
//           `${DATASET_DETAILS_URL}/${filename}`
//         );
//         if (!datasetRes.data) throw new Error("Dataset not found");
//         if (datasetRes.data.error) throw new Error(datasetRes.data.error);

//         setDataset(datasetRes.data.datastats);

//         const tasksRes = await axios.get(
//           `${TASKS_BY_DATASET_ID}/${datasetRes.data.dataset_id}`
//         );
//         if (tasksRes.data.error) throw new Error(tasksRes.data.error);

//         setTasks(tasksRes.data);
//       } catch (err) {
//         setError(err.response?.data?.error || err.message);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchData();
//   }, [filename]);

//   const fetchBenchmarks = async (taskId) => {
//     try {
//       const res = await axios.get(`${BENCHMARKS_BY_TASK_ID}/${taskId}`);
//       return res.data;
//     } catch (err) {
//       setError(err.response?.data?.detail || err.message);
//       return [];
//     }
//   };

//   const handleTaskSelect = async (task) => {
//     if (selectedTask?.task_id === task.task_id) {
//       setSelectedTask(null);
//     } else {
//       const benchmarks = await fetchBenchmarks(task.task_id);
//       setSelectedTask({ ...task, benchmarks });
//     }
//   };

//   if (loading)
//     return (
//       <div className="p-8 text-center">
//         <CogIcon className="h-12 w-12 animate-spin text-gray-400 mx-auto" />
//         <p className="mt-2 text-gray-600">Loading dataset details...</p>
//       </div>
//     );

//   if (error) return <div className="p-8 text-center text-red-600">{error}</div>;
//   if (!dataset)
//     return (
//       <div className="p-8 text-center text-gray-600">Dataset not found</div>
//     );

//   return (
//     <div className="max-w-7xl mx-auto p-6 space-y-8">
//       <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-6">
//         <SummaryStats
//           fileName={dataset.filename}
//           numRows={dataset.datastats?.numRows}
//           numCols={dataset.datastats?.numColumns}
//         />

//         {dataset.datastats?.columnStats && (
//           <ColumnCarousel columns={dataset.datastats.columnStats} />
//         )}
//       </div>

//       <div className="space-y-6">
//         <div className="flex justify-between items-center">
//           <h2 className="text-2xl font-semibold text-gray-800">
//             Associated Tasks {tasks.length > 0 && `(${tasks.length})`}
//           </h2>
//           <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 flex items-center gap-2">
//             <PlusIcon className="h-5 w-5" />
//             Create New Task
//           </button>
//         </div>

//         {tasks.length === 0 ? (
//           <div className="bg-white p-6 rounded-lg border border-gray-100 text-center text-gray-500">
//             No tasks found for this dataset
//           </div>
//         ) : (
//           tasks.map((task) => (
//             <TaskCard
//               key={task.task_id}
//               task={task}
//               isSelected={selectedTask?.task_id === task.task_id}
//               onSelect={() => handleTaskSelect(task)}
//             />
//           ))
//         )}
//       </div>

//       {selectedTask && (
//         <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 space-y-6">
//           <div className="flex justify-between items-center">
//             <h3 className="text-xl font-semibold">
//               {selectedTask.task_name} Metrics
//             </h3>
//             <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
//               {selectedTask.metric}
//             </span>
//           </div>

//           <LineChart
//             benchmarks={selectedTask.benchmarks}
//             metric={selectedTask.metric}
//           />
//           <BenchmarkList benchmarks={selectedTask.benchmarks} />
//         </div>
//       )}
//     </div>
//   );
// };

// export default ProcessedDataSetOverview;

import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import SummaryStats from "./DatasetDetails/SummaryStats";
import ColumnDetails from "./DatasetDetails/ColumnDetails";
import Tasks from "./DatasetDetails/TaskCard";

const PROCESSED_DATASET_URL = process.env.REACT_APP_PROCESSED_OVERVIEW_PATH;

const ProcessedDataSetOverview = () => {
  const { filename } = useParams();
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const response = await axios.get(
          `${PROCESSED_DATASET_URL}/${filename}`
        );
        setDataset(response.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [filename]);

  if (loading) return <div className="p-4">Loading dataset...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!dataset) return <div className="p-4">Dataset not found</div>;

  const columnDetails = {};
  dataset.datastats.columnStats.forEach((column) => {
    columnDetails[column.name] = column.type;
  });

  return (
    <div className="space-y-6">
      <SummaryStats
        fileName={dataset.filename}
        numRows={dataset.datastats?.numRows}
        numCols={dataset.datastats?.numColumns}
      />

      {dataset.datastats?.columnStats && (
        <ColumnDetails columnStats={dataset.datastats.columnStats} />
      )}

      <Tasks datasetId={dataset.dataset_id} />

      <PreprocessingDetails
        columns={columnDetails}
        fileName={dataset.filename}
        directory="Uploads"
      />
    </div>
  );
};

export default ProcessedDataSetOverview;
