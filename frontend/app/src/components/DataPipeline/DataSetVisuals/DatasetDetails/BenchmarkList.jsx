import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { useParams, Link } from "react-router-dom";
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
import {
  ClockIcon,
  ArrowTopRightOnSquareIcon,
} from "@heroicons/react/24/outline";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
);

const BENCHMARKS_URL = process.env.REACT_APP_GET_BENCHMARKS_WITH_TASK_ID;

const Benchmarks = () => {
  const { task_id } = useParams();
  const [benchmarks, setBenchmarks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedBenchmark, setSelectedBenchmark] = useState(null);
  const chartRef = useRef();

  useEffect(() => {
    const fetchBenchmarks = async () => {
      const response = await axios.get(`${BENCHMARKS_URL}/${task_id}`);
      setBenchmarks(response.data);
      setLoading(false);
    };
    fetchBenchmarks();
  }, [task_id]);

  const chartData = {
    datasets: [
      {
        label: "Metric Value",
        data: benchmarks.map((b) => ({
          x: new Date(b.created_at),
          y: b.metric_value,
        })),
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.5)",
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    onClick: (e) => {
      const chart = chartRef.current;
      if (chart) {
        const points = chart.getElementsAtEventForMode(
          e,
          "nearest",
          { intersect: true },
          true,
        );

        if (points.length > 0) {
          const firstPoint = points[0];
          const datasetIndex = firstPoint.datasetIndex;
          const index = firstPoint.index;
          setSelectedBenchmark(benchmarks[index]);
        }
      }
    },
    scales: {
      x: {
        type: "time",
        time: {
          unit: "day",
          tooltipFormat: "yyyy-MM-dd HH:mm",
        },
        grid: { display: false },
      },
      y: {
        beginAtZero: true,
        grid: { color: "#f3f4f6" },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (context) => new Date(context[0].parsed.x).toLocaleString(),
          label: (context) => `Score: ${context.parsed.y.toFixed(4)}`,
        },
      },
    },
  };

  return (
    <div className="p-4 space-y-6">
      <div className="bg-white p-4 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">
          Benchmarks for Task {task_id}
        </h2>

        <div className="space-y-6">
          <div className="relative h-64 w-full">
            <Line ref={chartRef} data={chartData} options={options} />
          </div>

          {selectedBenchmark && (
            <div className="border-t pt-4">
              <div className="flex items-center justify-between bg-blue-50 p-4 rounded-lg">
                <div className="flex items-center gap-4">
                  <div className="bg-blue-100 p-2 rounded-full">
                    <ClockIcon className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-semibold">
                      {selectedBenchmark.metric_value.toFixed(4)}
                    </p>
                    <p className="text-sm text-gray-600">
                      {new Date(
                        selectedBenchmark.created_at,
                      ).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <Link
                  to={`/benchmarks/${selectedBenchmark.benchmark_id}/training`}
                  target="_blank"
                  className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
                >
                  <span>Training Details</span>
                  <ArrowTopRightOnSquareIcon className="w-5 h-5" />
                </Link>
              </div>
            </div>
          )}

          {!loading && benchmarks.length === 0 && (
            <div className="text-center text-gray-500 py-6">
              No benchmarks found for this task
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Benchmarks;
