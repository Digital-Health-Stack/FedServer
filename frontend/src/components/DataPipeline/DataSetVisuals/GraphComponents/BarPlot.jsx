import React from "react";
import { Bar } from "react-chartjs-2";
import "chart.js/auto";

const trimLabel = (label) => {
  return label.length > 8 ? `${label.substring(0, 8)}...` : label;
};

const BarPlot = ({ bins, counts, xTitle = "Bins", yTitle = "Counts" }) => {
  const randomColors = counts.map(
    () =>
      `rgba(${Math.floor(Math.random() * 256)}, ${Math.floor(
        Math.random() * 256,
      )}, ${Math.floor(Math.random() * 256)}, 0.8)`,
  );

  const data = {
    labels: bins.map(trimLabel),
    datasets: [
      {
        label: "Count",
        data: counts,
        backgroundColor: randomColors,
        borderWidth: 0,
        categoryPercentage: 0.9,
        barPercentage: 0.99,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: {
          display: true,
          text: xTitle,
          font: { size: 14, weight: "bold" },
        },
        grid: { display: false },
        offset: false,
      },
      y: {
        title: {
          display: true,
          text: yTitle,
          font: { size: 14, weight: "bold" },
        },
        grid: { display: false },
        beginAtZero: true,
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: { enabled: true },
    },
    layout: {
      padding: {
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
      },
    },
  };

  return <Bar data={data} options={options} />;
};

export default BarPlot;
