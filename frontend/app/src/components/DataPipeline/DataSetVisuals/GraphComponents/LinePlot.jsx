// // LineChart Component
// import { Line } from 'react-chartjs-2';
// import { Chart as ChartJS,
//     CategoryScale,
//     LinearScale,
//     PointElement,
//     LineElement,
//     Title,
//     Tooltip,
//     Legend,
//     TimeScale
//   } from 'chart.js';
// import 'chartjs-adapter-date-fns';

// // Register Chart.js components
// ChartJS.register(
//     CategoryScale,
//     LinearScale,
//     PointElement,
//     LineElement,
//     Title,
//     Tooltip,
//     Legend,
//     TimeScale
//   );

// const LineChart = ({ benchmarks, metric }) => {
//     const chartData = {
//       datasets: [
//         {
//           label: metric,
//           data: benchmarks.map(b => ({
//             x: new Date(b.created_at),
//             y: b.metric_value
//           })),
//           borderColor: '#3b82f6',
//           backgroundColor: 'rgba(59, 130, 246, 0.5)',
//           tension: 0.1,
//           pointRadius: 5,
//           pointHoverRadius: 7
//         }
//       ]
//     };

//     const options = {
//       responsive: true,
//       maintainAspectRatio: false,
//       scales: {
//         x: {
//           type: 'time',
//           time: {
//             unit: 'day',
//             tooltipFormat: 'yyyy-MM-dd HH:mm'
//           },
//           title: {
//             display: true,
//             text: 'Date'
//           }
//         },
//         y: {
//           title: {
//             display: true,
//             text: metric
//           }
//         }
//       },
//       plugins: {
//         tooltip: {
//           callbacks: {
//             title: (context) =>
//               new Date(context[0].parsed.x).toLocaleString(),
//             label: (context) =>
//               `${metric}: ${context.parsed.y.toFixed(4)}`
//           }
//         }
//       }
//     };

//     return (
//       <div className="h-96 bg-white p-4 rounded-lg border border-gray-100">
//         <Line data={chartData} options={options} />
//       </div>
//     );
//   };
