import React, { useEffect, useState } from "react";
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

const BENCHMARKS_URL = process.env.REACT_APP_GET_BENCHMARKS_WITH_TASK_ID;

const Benchmarks = () => {
  const { task_id } = useParams();
  const [benchmarks, setBenchmarks] = useState([]);
  const [loading, setLoading] = useState(true);

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
        tension: 0.1,
      },
    ],
  };

  return (
    <div className="p-4 space-y-6">
      <div className="bg-white p-4 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">
          Benchmarks for Task {task_id}
        </h2>

        {!loading && benchmarks.length > 0 && (
          <div className="mb-6">
            <Line
              data={chartData}
              options={{
                responsive: true,
                scales: {
                  x: { type: "time", time: { unit: "day" } },
                  y: { beginAtZero: true },
                },
              }}
            />
          </div>
        )}

        <div className="space-y-3">
          {loading ? (
            <p>Loading benchmarks...</p>
          ) : benchmarks.length > 0 ? (
            benchmarks.map((benchmark) => (
              <div
                key={benchmark.benchmark_id}
                className="border p-3 rounded-lg hover:bg-gray-50"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">
                      {benchmark.metric_value.toFixed(4)}
                    </p>
                    <p className="text-sm text-gray-500">
                      {new Date(benchmark.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <Link
                    to={`/benchmarks/${benchmark.benchmark_id}/training`}
                    target="_blank"
                    className="text-blue-500 hover:text-blue-700 text-sm"
                  >
                    View Training Details →
                  </Link>
                </div>
              </div>
            ))
          ) : (
            <p className="text-gray-500">No benchmarks found</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Benchmarks;
