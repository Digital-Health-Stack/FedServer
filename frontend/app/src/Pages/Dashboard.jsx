import { useEffect, useState } from "react";
import { useAuth } from "../contexts/AuthContext";
// serverService.js

// Mock data for server stats
const mockStats = {
  total_sessions: 42,
  active_sessions: 5,
  completed_sessions: 35,
  pending_qpd_datasets: 3,
  registered_clients: 8,
  available_datasets: 12,
  avg_session_duration: 3540, // in seconds
  system_uptime: 86400, // in seconds (24 hours)
};


// Simulate API delay
const simulateApiDelay = () =>
  new Promise(
    (resolve) => setTimeout(resolve, Math.random() * 500 + 200) // 200-700ms delay
  );

export const getServerStats = async (api) => {
  await simulateApiDelay();
  return { data: mockStats };
};



// Optional: Add error simulation for testing error states
export const getServerStatsWithError = async (api) => {
  await simulateApiDelay();
  throw new Error("Failed to fetch server statistics");
};

export const getAllServerSessionsWithError = async (api) => {
  await simulateApiDelay();
  throw new Error("Failed to fetch server sessions");
};

export const listDatasetsWithError = async (api) => {
  await simulateApiDelay();
  throw new Error("Failed to fetch datasets");
};

export const getQpdDatasetsWithError = async (api) => {
  await simulateApiDelay();
  throw new Error("Failed to fetch QPD datasets");
};
import {
  ArrowPathIcon,
  CubeIcon,
  ArrowRightIcon,
  ClockIcon,
  ChartBarIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationCircleIcon,
  CircleStackIcon,
  ServerIcon,
  UsersIcon,
  BoltIcon,
  DocumentTextIcon,
  EyeIcon,
} from "@heroicons/react/24/outline";
import { getAllSessions } from "../services/federatedService";
import {
  getRawDatasets,
  getProcessedDatasets,
  listTransferredData,
} from "../services/privateService";
import { Link, useNavigate } from "react-router-dom";

export default function Dashboard() {
  const { api } = useAuth();
  const [stats, setStats] = useState(null);
  const [showDatasets, setShowDatasets] = useState("raw");
  const [sessions, setSessions] = useState([]);
  const [datasets, setDatasets] = useState({ uploads: [], processed: [] });
  const [qpdDatasets, setQpdDatasets] = useState([]);
  const [loading, setLoading] = useState({
    stats: true,
    sessions: true,
    datasets: true,
    qpd: true,
  });
  const [error, setError] = useState({
    stats: null,
    sessions: null,
    datasets: null,
    qpd: null,
  });
  const navigate = useNavigate();

  // Status mapping
  const statusMap = {
    0: { text: "Created", color: "bg-gray-100 text-gray-800", icon: CubeIcon },
    1: {
      text: "Negotiation",
      color: "bg-yellow-100 text-yellow-800",
      icon: ExclamationCircleIcon,
    },
    2: {
      text: "Recruitment",
      color: "bg-blue-100 text-blue-800",
      icon: UsersIcon,
    },
    3: {
      text: "Initializing",
      color: "bg-indigo-100 text-indigo-800",
      icon: ServerIcon,
    },
    4: {
      text: "Training",
      color: "bg-purple-100 text-purple-800",
      icon: BoltIcon,
    },
    5: {
      text: "Completed",
      color: "bg-green-100 text-green-800",
      icon: CheckCircleIcon,
    },
    [-1]: {
      text: "Failed",
      color: "bg-red-100 text-red-800",
      icon: XCircleIcon,
    },
  };

  // Fetch server statistics
  const fetchStats = async () => {
    try {
      setLoading((prev) => ({ ...prev, stats: true }));
      const response = await getServerStats(api);
      setStats(response.data);
    } catch (err) {
      setError((prev) => ({
        ...prev,
        stats: "Failed to load server statistics",
      }));
      console.error("Error fetching stats:", err);
    } finally {
      setLoading((prev) => ({ ...prev, stats: false }));
    }
  };

  // Fetch all server sessions
  const fetchSessions = async () => {
    try {
      setLoading((prev) => ({ ...prev, sessions: true }));
      console.log("Checkpoint 1");
      // Use the new getAllSessions function with pagination
      const response = await getAllSessions(api, 1, 4); // page = 1, perPage = 6
      setSessions(response.data.data);
      setError((prev) => ({ ...prev, sessions: null }));
    } catch (err) {
      setError((prev) => ({ ...prev, sessions: "Failed to load sessions" }));
      console.error("Error fetching sessions:", err);
    } finally {
      setLoading((prev) => ({ ...prev, sessions: false }));
    }
  };

  // Fetch available datasets
  const fetchDatasets = async () => {
    try {
      const [raw, processed] = await Promise.all([
        getRawDatasets().catch(() => ({ data: [] })), // Handle rejected promises
        getProcessedDatasets().catch(() => ({ data: [] })),
      ]);

      const uploads = Array.isArray(raw.data) ? raw.data.slice(0, 4) : [];
      const processedData = Array.isArray(processed.data)
        ? processed.data.slice(0, 4)
        : [];

      setDatasets({ uploads, processed: processedData });
    } catch (error) {
      setDatasets({ uploads: [], processed: [] });
      console.error("Error fetching datasets:", error);
    }
  };

  // Fetch QPD datasets
  const fetchQpdDatasets = async () => {
    try {
      setLoading((prev) => ({ ...prev, qpd: true }));
      const params = {
        skip: 0,
        limit: 4,
      };
      const response = await listTransferredData(params);
      console.log(response.data);
      setQpdDatasets(response.data); // Show latest 5 QPD datasets
    } catch (err) {
      setError((prev) => ({ ...prev, qpd: "Failed to load QPD datasets" }));
      console.error("Error fetching QPD datasets:", err);
    } finally {
      setLoading((prev) => ({ ...prev, qpd: false }));
    }
  };

  // Format timestamp
  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  // Refresh all data
  const refreshAll = () => {
    fetchStats();
    fetchSessions();
    fetchDatasets();
    fetchQpdDatasets();
  };
  const handleSessionClick = (sessionId) => {
    navigate(`/trainings/${sessionId}`);
  };

  useEffect(() => {
    refreshAll();
  }, []);
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex justify-between items-center pb-6 border-b border-gray-200">
          <div className="flex items-center space-x-4">
            <div className="p-2 rounded-lg bg-indigo-100">
              <ServerIcon className="h-6 w-6 text-indigo-600" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">
                FedServer
              </h1>
              <p className="text-sm text-gray-500">Administration Dashboard</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="hidden md:block text-right">
              <p className="text-xs font-medium text-gray-500">Last updated</p>
              <p className="text-sm text-gray-900">
                {new Date().toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </p>
            </div>
            <button
              onClick={refreshAll}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <ArrowPathIcon className="h-4 w-4 mr-2 -ml-1" />
              Refresh Data
            </button>
          </div>
        </div>
        {/* Stats Grid with better spacing--- Hardcoded for now */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Total Sessions */}
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <div className="bg-indigo-100 p-3 rounded-lg mr-4">
              <CubeIcon className="h-5 w-5 text-indigo-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500">Total Sessions</p>
              <p className="text-base font-semibold mt-1">
                {loading.stats ? "--" : stats?.total_sessions || 0}
              </p>
            </div>
          </div>

          {/* Active Sessions */}
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <div className="bg-purple-100 p-3 rounded-lg mr-4">
              <BoltIcon className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500">Active</p>
              <p className="text-base font-semibold mt-1">
                {loading.stats ? "--" : stats?.active_sessions || 0}
              </p>
            </div>
          </div>

          {/* Completed Sessions */}
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <div className="bg-green-100 p-3 rounded-lg mr-4">
              <CheckCircleIcon className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500">Completed</p>
              <p className="text-base font-semibold mt-1">
                {loading.stats ? "--" : stats?.completed_sessions || 0}
              </p>
            </div>
          </div>

          {/* QPD Datasets */}
          <div className="bg-white p-4 rounded-lg shadow-sm flex items-center">
            <div className="bg-blue-100 p-3 rounded-lg mr-4">
              <CircleStackIcon className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <p className="text-xs text-gray-500">Pending QPD</p>
              <p className="text-base font-semibold mt-1">
                {loading.stats ? "--" : stats?.pending_qpd_datasets || 0}
              </p>
            </div>
          </div>
        </div>

        {/* Main Content with balanced spacing */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Sessions */}
          <div className="bg-white rounded-lg shadow-sm p-5 lg:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-900">
                Recent Sessions
              </h2>
              <button
                className="text-indigo-600 text-xs hover:underline"
                onClick={() => navigate("/trainings")}
              >
                View All →
              </button>
            </div>

            {loading.sessions ? (
              <div className="flex justify-center py-6">
                <ArrowPathIcon className="h-5 w-5 text-indigo-500 animate-spin" />
              </div>
            ) : error.sessions ? (
              <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-xs text-red-700">
                {error.sessions}
              </div>
            ) : (
              <div className="space-y-3">
                {sessions.slice(0, 4).map((session) => (
                  <div
                    key={session.id}
                    onClick={() => handleSessionClick(session.id)}
                    className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center">
                      <div
                        className={`h-3 w-3 rounded-full mr-3 ${
                          statusMap[session.training_status]?.color.split(
                            " "
                          )[0] || "bg-gray-300"
                        }`}
                      />
                      <div>
                        <p className="text-sm font-medium text-gray-900 truncate w-40">
                          {session.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          ID: {session.id}
                        </p>
                      </div>
                    </div>
                    <span
                      className={`text-xs px-2 py-1 rounded ${
                        statusMap[session.training_status]?.color ||
                        "bg-gray-100 text-gray-800"
                      }`}
                    >
                      {statusMap[session.training_status]?.text || "Unknown"}
                    </span>
                  </div>
                ))}
                {sessions.length === 0 && (
                  <div className="text-center py-4 text-sm text-gray-500">
                    No active sessions
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Server Stats ---- Hardcoded for now*/}
          <div className="bg-white rounded-lg shadow-sm p-5">
            <h2 className="text-sm font-semibold text-gray-900 mb-4">
              Server Health & Status
            </h2>

            {loading.stats ? (
              <div className="flex justify-center py-6">
                <ArrowPathIcon className="h-5 w-5 text-indigo-500 animate-spin" />
              </div>
            ) : error.stats ? (
              <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-xs text-red-700">
                {error.stats}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">
                    Active Training Sessions:
                  </span>
                  <span className="text-sm font-medium">
                    {stats?.active_sessions || 0}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">
                    Pending Client Updates:
                  </span>
                  <span className="text-sm font-medium">
                    {stats?.pending_updates || 0}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">
                    Storage Utilization:
                  </span>
                  <span className="text-sm font-medium">
                    {stats?.storage_used_gb
                      ? `${stats.storage_used_gb}GB / ${stats.storage_total_gb}GB`
                      : "--"}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">
                    Last Model Accuracy:
                  </span>
                  <span className="text-sm font-medium">
                    {stats?.last_accuracy
                      ? `${(stats.last_accuracy * 100).toFixed(1)}%`
                      : "--"}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">Avg Round Time:</span>
                  <span className="text-sm font-medium">
                    {stats?.avg_round_time ? `${stats.avg_round_time}s` : "--"}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Datasets Section with improved spacing */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Available Datasets */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="mb-6">
              {/* First row - Title and View All */}
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xl font-semibold text-gray-800">
                  Recent Datasets
                </h2>
                <Link
                  to="/ManageData"
                  className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
                >
                  View all
                  <ArrowRightIcon className="ml-1 h-4 w-4" />
                </Link>
              </div>

              {/* Second row - Filter */}
              <div className="flex justify-end items-center gap-2">
                <span className="text-sm font-medium text-gray-600">
                  Filter:
                </span>
                <div className="inline-flex bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setShowDatasets("raw")}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      showDatasets === "raw"
                        ? "bg-white shadow-sm text-blue-600"
                        : "text-gray-600 hover:text-gray-800"
                    }`}
                  >
                    Raw Data
                  </button>
                  <button
                    onClick={() => setShowDatasets("processed")}
                    className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                      showDatasets === "processed"
                        ? "bg-white shadow-sm text-green-600"
                        : "text-gray-600 hover:text-gray-800"
                    }`}
                  >
                    Processed
                  </button>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              {(showDatasets === "raw" ? datasets.uploads : datasets.processed)
                .slice(0, 4)
                .map((dataset) => (
                  <div
                    key={dataset.filename}
                    className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      <div
                        className={`p-2 rounded-lg ${
                          showDatasets === "raw"
                            ? "bg-blue-100"
                            : "bg-green-100"
                        }`}
                      >
                        {showDatasets === "raw" ? (
                          <DocumentTextIcon className="w-5 h-5 text-blue-600" />
                        ) : (
                          <CheckCircleIcon className="w-5 h-5 text-green-600" />
                        )}
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900 truncate max-w-[180px]">
                          {dataset.filename}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(dataset.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <Link
                      to={`/${
                        showDatasets === "raw" ? "raw" : "processed"
                      }-dataset-overview/${dataset.filename}`}
                      className="text-gray-400 hover:text-gray-600"
                      title="View details"
                    >
                      <EyeIcon className="w-5 h-5" />
                    </Link>
                  </div>
                ))}

              {((showDatasets === "raw" && !datasets.uploads.length) ||
                (showDatasets === "processed" &&
                  !datasets.processed.length)) && (
                <div className="text-center py-4 text-gray-500">
                  No {showDatasets} datasets found
                </div>
              )}
            </div>
          </div>

          {/* QPD Datasets */}
          <div className="bg-white rounded-lg shadow-sm p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-900">
                Pending QPD
              </h2>
              <button className="text-indigo-600 text-xs hover:underline" onClick={() => navigate("/assess-data-quality")}>
                View All →
              </button>
            </div>

            {loading.qpd ? (
              <div className="flex justify-center py-6">
                <ArrowPathIcon className="h-5 w-5 text-indigo-500 animate-spin" />
              </div>
            ) : error.qpd ? (
              <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-xs text-red-700">
                {error.qpd}
              </div>
            ) : (
              <div className="space-y-3">
                {qpdDatasets.slice(0, 4).map((qpd) => (
                  <div
                    key={qpd.id}
                    className="group flex justify-between items-start p-3 hover:bg-gray-50 rounded-lg"
                  >
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate max-w-[180px]">
                        {qpd.training_name}
                      </p>
                      <div className="flex flex-wrap items-center mt-1 gap-x-4 gap-y-1">
                        <span className="text-xs text-gray-500">
                          {qpd.num_datapoints} data points
                        </span>
                        <span className="text-xs text-gray-500">
                          Parent: {qpd.parent_filename}
                        </span>
                        <span className="text-xs text-gray-400">
                          Transferred:{" "}
                          {new Date(qpd.transferredAt).toLocaleDateString()}
                        </span>
                        <span className="text-xs text-gray-400">
                          Session ID: {qpd.federated_session_id}
                        </span>
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 self-center">
                      {qpd.approvedAt ? (
                        <span className="text-green-600">Approved</span>
                      ) : (
                        <span className="text-yellow-600">Pending</span>
                      )}
                    </div>
                  </div>
                ))}
                {qpdDatasets.length === 0 && (
                  <div className="text-center py-4 text-sm text-gray-500">
                    No pending QPD datasets
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
