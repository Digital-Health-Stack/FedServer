import React, { useState } from "react";
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
} from "chart.js";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Helper function to generate random color with optional alpha value
const getRandomColor = (alpha = 1) => {
  const r = Math.floor(Math.random() * 256);
  const g = Math.floor(Math.random() * 256);
  const b = Math.floor(Math.random() * 256);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const MetricsChart = ({ testResults }) => {
  const labels = Object.keys(testResults);
  const [selectedMetric, setSelectedMetric] = useState("");

  // Extract all unique metrics
  const metricsSet = new Set();
  labels.forEach((round) => {
    Object.keys(testResults[round]).forEach((metric) => {
      metricsSet.add(metric);
    });
  });
  const metrics = Array.from(metricsSet);

  // Prepare data for each metric
  const datasets = selectedMetric
    ? [
        {
          label: selectedMetric,
          data: labels.map((round) => testResults[round][selectedMetric]),
          fill: false,
          borderColor: getRandomColor(0.4),
          borderWidth: 1,
          tension: 0,
        },
      ]
    : [];

  const data = {
    labels: labels,
    datasets: datasets,
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            return `${context.dataset.label}: ${context.raw.toFixed(3)}`;
          },
        },
      },
    },
  };

  return (
    <div className="d-flex justify-content-center mb-5 bg-light">
      <div className="card mt-4" style={{ width: "60%" }}>
        <div className="card-body">
          <h4 className="text-center bg-dark text-white card-header">
            Metrics Overview
          </h4>
          <div className="d-flex align-items-center mb-3 mt-3">
            <label htmlFor="metricSelect" className="form-label me-2 mb-0">
              Select Metric:
            </label>
            <select
              id="metricSelect"
              className="form-select me-3"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              <option value="">Select a Metric</option>
              {metrics.map((metric) => (
                <option key={metric} value={metric}>
                  {metric.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
          <div className="d-flex justify-content-center">
            <div style={{ width: "100%" }}>
              <Line data={data} options={options} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsChart;
