import React from "react";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  Filler,
} from "chart.js";

// for performace reasons, i'm importing only the necessary components otherwise import the whole chart.js library for ease
// import "chart.js/auto";

// Register necessary chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  Filler
);

const HistogramPlot = ({
  bins,
  counts,
  xTitle = "Bins",
  yTitle = "Counts",
}) => {
  // Generate random colors for each bin
  const randomColors = counts.map(
    () =>
      `rgba(${Math.floor(Math.random() * 256)}, ${Math.floor(
        Math.random() * 256
      )}, ${Math.floor(Math.random() * 256)}, 0.8)`
  );

  // Formatting bins to show ranges with 2 decimal places
  const formattedBins = bins.map((bin, index) => {
    const nextBin = bins[index + 1];
    return nextBin
      ? `${bin.toFixed(2)}-${nextBin.toFixed(2)}`
      : `${bin.toFixed(2)}+`; // Last bin with a "+" sign
  });

  const data = {
    labels: formattedBins,
    datasets: [
      {
        label: "count",
        data: counts,
        borderColor: "rgba(0, 0, 0,0.5)",
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: "rgba(215, 9, 20, 0.5)",
        pointHoverRadius: 6,
        pointHoverBackgroundColor: "rgb(25, 39, 240)",
        type: "line",
      },
      {
        label: "Count",
        data: counts,
        backgroundColor: randomColors,
        borderWidth: 0,
        barPercentage: 1,
        categoryPercentage: 1,
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
        grid: {
          display: false,
        },
      },
      y: {
        ticks: {
          display: true,
        },
        title: {
          display: true,
          text: yTitle,
          font: { size: 14, weight: "bold" },
        },
        grid: {
          display: false,
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
      },
    },
    animation: {
      duration: 1500,
      easing: "easeOutBounce",
    },
    elements: {
      bar: {
        borderWidth: 2,
        borderColor: "rgba(0, 0, 0, 0.3)",
      },
    },

    // Custom callback to draw the counts above the bars
    onAfterDraw: (chart) => {
      const ctx = chart.ctx;
      const dataset = chart.data.datasets[0];

      dataset.data.forEach((value, index) => {
        const x = chart.getDatasetMeta(0).data[index].x;
        const y = chart.getDatasetMeta(0).data[index].y;
        const width = chart.getDatasetMeta(0).data[index].width;
        const height = chart.getDatasetMeta(0).data[index].height;

        // Draw the count above the bar
        ctx.font = "bold 14px sans-serif";
        ctx.fillStyle = "black";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom"; // Position the text just above the bar
        ctx.fillText(value, x, y - height / 2); // Adjust position slightly above the bar
      });
    },
  };

  return (
    <div className="w-full h-96">
      <Bar data={data} options={options} />
    </div>
  );
};

export default HistogramPlot;
