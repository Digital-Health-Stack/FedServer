import React, { useEffect, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { getTrainingResults } from "../../services/federatedService";
import {
  ArrowPathIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChartBarIcon,
} from "@heroicons/react/24/solid";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const Result = ({ sessionId }) => {
  const [results, setResults] = useState({
    server_results: {},
    client_results: {},
    test_metrics: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("chart");
  const [selectedMetric, setSelectedMetric] = useState(""); // Now single metric selection
  const { api } = useAuth();

  const fetchResultsData = async () => {
    if (!sessionId) {
      setResults({
        server_results: {},
        client_results: {},
        test_metrics: []
      });
      setLoading(false);
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await getTrainingResults(api, sessionId);
      setResults(response.data);
      
      // Set first metric as default selection if available
      if (response.data.test_metrics?.length > 0) {
        setSelectedMetric(response.data.test_metrics[0]);
      }
    } catch (err) {
      console.error("Error fetching training results:", err);
      setError("Failed to fetch training results. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResultsData();
  }, [sessionId]);

  const formatMetricValue = (value) => {
    return typeof value === 'number' ? value.toFixed(4) : value;
  };

  // Prepare data for charts for the selected metric
  const prepareChartData = () => {
    if (!selectedMetric) return [];
    
    const chartData = [];
    const rounds = Object.keys(results.server_results[selectedMetric] || {});

    rounds.forEach(round => {
      const roundNumber = parseInt(round.split('_')[1]);
      const roundData = { 
        round: `Round ${roundNumber}`,
        [`server_${selectedMetric}`]: results.server_results[selectedMetric]?.[round],
        [`client_${selectedMetric}`]: results.client_results[selectedMetric]?.[round]
      };

      chartData.push(roundData);
    });

    return chartData;
  };

  const chartData = prepareChartData();

  return (
    <div className="w-full space-y-6">
      <div className="bg-white shadow rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h3 className="text-lg font-medium text-gray-900 flex items-center">
            <ChartBarIcon className="h-5 w-5 mr-2 text-blue-500" />
            Training Results
            {loading && (
              <ArrowPathIcon className="h-5 w-5 ml-2 text-blue-500 animate-spin" />
            )}
          </h3>
          
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveTab("chart")}
              className={`px-3 py-1 text-sm rounded-md ${
                activeTab === "chart"
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              Chart View
            </button>
            <button
              onClick={() => setActiveTab("table")}
              className={`px-3 py-1 text-sm rounded-md ${
                activeTab === "table"
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              Table View
            </button>
          </div>
        </div>

        <div className="p-4">
          {activeTab === "table" ? (
            <div className="overflow-hidden">
              <div className="flex flex-wrap gap-2 mb-4">
                {results.test_metrics.map((metric) => (
                  <button
                    key={metric}
                    onClick={() => setSelectedMetric(metric)}
                    className={`px-3 py-1 text-sm rounded-md capitalize ${
                      selectedMetric === metric
                        ? "bg-blue-100 text-blue-700"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {metric}
                  </button>
                ))}
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Round
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Server {selectedMetric?.toUpperCase()}
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Your {selectedMetric?.toUpperCase()}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {loading ? (
                      <tr>
                        <td colSpan={3} className="px-6 py-4 text-center">
                          <div className="flex justify-center items-center text-gray-500">
                            <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                            Loading results...
                          </div>
                        </td>
                      </tr>
                    ) : error ? (
                      <tr>
                        <td colSpan={3} className="px-6 py-4 text-center">
                          <div className="flex justify-center items-center text-red-500">
                            <ExclamationTriangleIcon className="h-5 w-5 mr-2" />
                            {error}
                          </div>
                        </td>
                      </tr>
                    ) : !selectedMetric || Object.keys(results.server_results).length === 0 ? (
                      <tr>
                        <td colSpan={3} className="px-6 py-4 text-center">
                          <div className="flex justify-center items-center text-gray-500">
                            <InformationCircleIcon className="h-5 w-5 mr-2" />
                            {sessionId
                              ? "No results available yet"
                              : "Select a session to view results"}
                          </div>
                        </td>
                      </tr>
                    ) : (
                      Object.keys(results.server_results[selectedMetric] || {}).map(round => {
                        const roundNumber = round.split('_')[1];
                        return (
                          <tr key={round} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              Round {roundNumber}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {formatMetricValue(results.server_results[selectedMetric]?.[round])}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {results.client_results[selectedMetric]?.[round] 
                                ? formatMetricValue(results.client_results[selectedMetric][round])
                                : "-"}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex flex-wrap gap-2 mb-4">
                {results.test_metrics.map((metric) => (
                  <button
                    key={metric}
                    onClick={() => setSelectedMetric(metric)}
                    className={`px-3 py-1 text-sm rounded-md capitalize ${
                      selectedMetric === metric
                        ? "bg-blue-100 text-blue-700"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {metric}
                  </button>
                ))}
              </div>

              {loading ? (
                <div className="flex justify-center items-center h-64 text-gray-500">
                  <ArrowPathIcon className="h-8 w-8 mr-2 animate-spin" />
                  Loading charts...
                </div>
              ) : error ? (
                <div className="flex justify-center items-center h-64 text-red-500">
                  <ExclamationTriangleIcon className="h-8 w-8 mr-2" />
                  {error}
                </div>
              ) : !selectedMetric || Object.keys(results.server_results).length === 0 ? (
                <div className="flex justify-center items-center h-64 text-gray-500">
                  <InformationCircleIcon className="h-8 w-8 mr-2" />
                  {sessionId
                    ? "No results available for charts"
                    : "Select a session to view charts"}
                </div>
              ) : (
                <div className="h-80">
                  <h4 className="text-md font-medium text-gray-700 mb-2">
                    {selectedMetric.toUpperCase()} Progression
                  </h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip formatter={(value) => [formatMetricValue(value)]} />
                      <Legend />
                      <Line
                        name={`Server ${selectedMetric.toUpperCase()}`}
                        type="monotone"
                        dataKey={`server_${selectedMetric}`}
                        stroke="#3b82f6" // Blue
                        activeDot={{ r: 8 }}
                      />
                      {Object.keys(results.client_results[selectedMetric] || {}).length > 0 && (
                        <Line
                          name={`Your ${selectedMetric.toUpperCase()}`}
                          type="monotone"
                          dataKey={`client_${selectedMetric}`}
                          stroke="#10b981" // Green
                          activeDot={{ r: 8 }}
                          strokeDasharray="5 5"
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="px-6 py-3 bg-gray-50 text-right text-xs text-gray-500 border-t border-gray-200">
          {sessionId && `Session ID #${sessionId} | Current Round: ${results.current_round || 'N/A'}`}
        </div>
      </div>
    </div>
  );
};

export default Result;