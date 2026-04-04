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
    (resolve) => setTimeout(resolve, Math.random() * 500 + 200), // 200-700ms delay
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
  InformationCircleIcon,
} from "@heroicons/react/24/outline";
import { getAllSessions, getSessionStats } from "../services/federatedService";
import { getAllDatasets, listTransferredData } from "../services/privateService";
import { Link, useNavigate } from "react-router-dom";

export default function Dashboard() {
  const { api } = useAuth();
  const [stats, setStats] = useState(null);
  const [showDatasets, setShowDatasets] = useState("raw");
  const [sessions, setSessions] = useState([]);
  const [sessStats, setSessionStats] = useState([]);
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
    CREATED: {
      text: "Created",
      color: "bg-gray-100 text-gray-800",
      icon: CubeIcon,
    },
    PRICE_NEGOTIATION: {
      text: "Negotiation",
      color: "bg-yellow-100 text-yellow-800",
      icon: ExclamationCircleIcon,
    },
    ACCEPTING_CLIENTS: {
      text: "Recruitment",
      color: "bg-blue-100 text-blue-800",
      icon: UsersIcon,
    },
    MODEL_INITIALIZATION: {
      text: "Initializing",
      color: "bg-indigo-100 text-indigo-800",
      icon: ServerIcon,
    },
    STARTED: {
      text: "Training",
      color: "bg-purple-100 text-purple-800",
      icon: BoltIcon,
    },
    COMPLETED: {
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
      console.log(response.data.data.length);

      // Print first 5 sessions in console
      // response.data.data.forEach((session, index) => {
      //   console.log(
      //     `#${index + 1} → ID: ${session.id}, Name: ${session.name}, Status: ${session.training_status}`,
      //   );
      // });

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
      const res = await getAllDatasets(0, 8).catch(() => ({ data: { datasets: [] } }));
      const list = res.data?.datasets ?? [];
      const slice = Array.isArray(list) ? list.slice(0, 4) : [];
      setDatasets({ uploads: slice, processed: slice });
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

  const fetchSessionsStats = async () => {
    try {
      setLoading((prev) => ({ ...prev, sessions: true }));
      console.log("Fetching session stats...");

      const response = await getSessionStats(api);

      // Directly use the stats from backend
      const stats = response.data;
      console.log("Session stats:", stats);

      // Optionally set state
      setSessionStats(stats);

      setError((prev) => ({ ...prev, sessions: null }));
    } catch (err) {
      setError((prev) => ({ ...prev, sessions: "Failed to load sessions" }));
      console.error("Error fetching sessions:", err);
    } finally {
      setLoading((prev) => ({ ...prev, sessions: false }));
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
    fetchSessionsStats();
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
          <div className="bg-white p-4 rounded-xl border-2 border-indigo-300 shadow-sm flex items-center">
            <div className="bg-indigo-100 p-3 rounded-lg mr-4">
              <CubeIcon className="h-5 w-5 text-indigo-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-800">
                Total Sessions
              </p>
              <p className="text-base font-semibold mt-1">
                {loading.sessStats ? "--" : sessStats?.total_sessions || 0}
              </p>
            </div>
          </div>

          {/* Active Sessions */}
          <div className="bg-white p-4 rounded-xl border-2 border-indigo-300 shadow-sm flex items-center">
            <div className="bg-purple-100 p-3 rounded-lg mr-4">
              <BoltIcon className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-800">Active</p>
              <p className="text-base font-semibold mt-1">
                {loading.sessStats ? "--" : sessStats?.active_sessions || 0}
              </p>
            </div>
          </div>

          {/* Completed Sessions */}
          <div className="bg-white p-4 rounded-xl border-2 border-indigo-300 shadow-sm flex items-center">
            <div className="bg-green-100 p-3 rounded-lg mr-4">
              <CheckCircleIcon className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-800">Completed</p>
              <p className="text-base font-semibold mt-1">
                {loading.sessStats ? "--" : sessStats?.completed_sessions || 0}
              </p>
            </div>
          </div>

          {/* QPD Datasets */}
          <div className="bg-white p-4 rounded-xl border-2 border-indigo-300 shadow-sm flex items-center">
            <div className="bg-blue-100 p-3 rounded-lg mr-4">
              <CircleStackIcon className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-800">Pending QPD</p>
              <p className="text-base font-semibold mt-1">
                {loading.stats ? "--" : stats?.pending_qpd_datasets || 0}
              </p>
            </div>
          </div>
        </div>

        {/* Main Content with balanced spacing */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Sessions */}
          {/* <div className="bg-white rounded-xl border-2 border-indigo-300 shadow-sm p-5 lg:col-span-2">
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
                            " ",
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
          </div> */}
          {/* bg-white rounded-xl border-2 border-indigo-300 shadow-sm p-5 lg:col-span-2 */}
          <div className="bg-white rounded-xl shadow-sm border-2 border-indigo-300 lg:col-span-2 overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 px-6 py-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {/* Icon */}
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <svg
                      className="w-5 h-5 text-blue-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  </div>

                  {/* Title + Subtitle */}
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">
                      Recent Sessions
                    </h2>
                    <p className="text-sm text-gray-500">
                      {sessions.length} session
                      {sessions.length !== 1 ? "s" : ""} available
                    </p>
                  </div>
                </div>

                {/* View All Button */}
                <button
                  onClick={() => navigate("/trainings")}
                  className="inline-flex items-center px-3 py-2 text-sm font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 transition-colors"
                >
                  View All
                  <svg
                    className="ml-1 w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </button>
              </div>
            </div>

            {/* Body */}
            {/* <div className="p-5">
              {loading.sessions ? (
                <div className="flex justify-center py-6">
                  <ArrowPathIcon className="h-5 w-5 text-blue-500 animate-spin" />
                </div>
              ) : error.sessions ? (
                <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-sm text-red-700">
                  {error.sessions}
                </div>
              ) : (
                <div className="space-y-3">
                  {console.log(sessions)}
                  {sessions.slice(0, 4).map((session) => (
                    <div
                      key={session.id}
                      onClick={() => handleSessionClick(session.id)}
                      className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg transition cursor-pointer"
                    >
                      <div className="flex items-center">
                        <div
                          className={`h-3 w-3 rounded-full mr-3 ${
                            statusMap[session.training_status]?.color?.split(
                              " ",
                            )[0] || "bg-gray-300"
                          }`}
                        />
                        <div>
                          <p className="text-sm font-medium text-gray-900 truncate w-44">
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
                        {statusMap[session.training_status]?.text || "ye le mc"}
                      </span>
                    </div>
                  ))}

                  {sessions.length === 0 && (
                    <div className="text-center py-6 text-sm text-gray-500">
                      No active sessions
                    </div>
                  )}
                </div>
              )}
            </div> */}
            <div className="p-5">
              {loading.sessions ? (
                <div className="flex justify-center py-6">
                  <ArrowPathIcon className="h-5 w-5 text-blue-500 animate-spin" />
                </div>
              ) : error.sessions ? (
                <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-sm text-red-700">
                  {error.sessions}
                </div>
              ) : (
                <div className="space-y-3">
                  {sessions.slice(0, 4).map((session) => {
                    let icon, bgColor, textColor, text;
                    switch (session.training_status) {
                      case "CREATED":
                        icon = (
                          <svg
                            className="w-5 h-5 text-gray-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                            />
                          </svg>
                        );
                        bgColor = "bg-gray-100";
                        textColor = "text-gray-800";
                        text = "Created";
                        break;
                      case "PRICE_NEGOTIATION":
                        icon = (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="w-5 h-5"
                            viewBox="0 0 72 72"
                          >
                            <ellipse
                              cx="34.153"
                              cy="34.635"
                              fill="#fcea2b"
                              rx="29.033"
                              ry="22.118"
                              transform="rotate(-51.131 34.153 34.635)"
                            />
                            <path
                              fill="#fff"
                              d="M52.139 12.03a18.698 18.698 0 0 0-13.3-3.86a17.98 17.98 0 0 1 10.077 3.86c9.51 7.666 9.064 24-.998 36.484c-6.14 7.618-14.44 12.034-22.14 12.584c8.53.639 18.353-3.889 25.362-12.584c10.062-12.484 10.51-28.819.999-36.484Z"
                            />
                            <path
                              fill="#f1b31c"
                              d="M55.349 13.082a20.374 20.374 0 0 0-1.558-1.138a.928.928 0 0 0-.11-.045a1.03 1.03 0 0 0-.556-.102a.923.923 0 0 0-.527.235a.943.943 0 0 0-.094.069c-.019.02-.025.047-.042.068s-.041.034-.057.057a.945.945 0 0 0-.044.11a1.027 1.027 0 0 0-.102.562a.978.978 0 0 0 .043.183a.834.834 0 0 0 .19.336a.941.941 0 0 0 .07.097c8.278 7.636 7.415 22.736-1.965 34.375c-9.702 12.035-24.97 15.88-34.034 8.573a.96.96 0 0 0-.187-.097l-.055-.029a.97.97 0 0 0-.3-.074c-.027-.002-.053-.006-.08-.007a.916.916 0 0 0-.59.196a.94.94 0 0 0-.098.058c-.02.017-.027.041-.044.06c-.016.017-.038.025-.053.043a.935.935 0 0 0-.047.091a.96.96 0 0 0-.117.226a.934.934 0 0 0-.024.097a.972.972 0 0 0-.026.297c.002.02.006.038.009.058a.965.965 0 0 0 .096.312c.008.016.018.03.027.045a.953.953 0 0 0 .109.183a20.069 20.069 0 0 0 2.286 2.158a21.13 21.13 0 0 0 13.441 4.555c8.59 0 17.89-4.48 24.528-12.715c10.444-12.957 10.403-30.38-.09-38.837Z"
                            />
                          </svg>
                        );
                        bgColor = "bg-yellow-100";
                        textColor = "text-yellow-800";
                        text = "Negotiation";
                        break;
                      case "ACCEPTING_CLIENTS":
                        icon = (
                          <svg
                            className="w-5 h-5 text-blue-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                            />
                          </svg>
                        );
                        bgColor = "bg-blue-100";
                        textColor = "text-blue-800";
                        text = "Accepting Clients";
                        break;
                      case "MODEL_INITIALIZATION":
                        icon = (
                          <svg
                            className="w-5 h-5 text-indigo-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                            />
                          </svg>
                        );
                        bgColor = "bg-indigo-100";
                        textColor = "text-indigo-800";
                        text = "Initializing";
                        break;
                      case "STARTED":
                        icon = (
                          <svg
                            className="w-5 h-5 text-purple-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M13 10V3L4 14h7v7l9-11h-7z"
                            />
                          </svg>
                        );
                        bgColor = "bg-purple-100";
                        textColor = "text-purple-800";
                        text = "Training Active";
                        break;
                      case "COMPLETED":
                        icon = (
                          <svg
                            className="w-5 h-5 text-green-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        );
                        bgColor = "bg-green-100";
                        textColor = "text-green-800";
                        text = "Completed";
                        break;
                      case "FAILED":
                        icon = (
                          <svg
                            className="w-5 h-5 text-red-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L4.34 16.5c-.77.833.192 2.5 1.732 2.5z"
                            />
                          </svg>
                        );
                        bgColor = "bg-red-100";
                        textColor = "text-red-800";
                        text = "Failed";
                        break;
                      case "CANCELLED":
                        icon = (
                          <svg
                            className="w-5 h-5 text-red-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        );
                        bgColor = "bg-red-100";
                        textColor = "text-red-800";
                        text = "Cancelled";
                        break;
                      default:
                        icon = (
                          <svg
                            className="w-5 h-5 text-gray-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        );
                        bgColor = "bg-gray-100";
                        textColor = "text-gray-800";
                        text = "Unknown";
                    }

                    return (
                      <div
                        key={session.id}
                        className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg transition"
                      >
                        <div className="flex items-center space-x-3">
                          {/* Icon */}
                          <div className={`p-2 rounded-lg ${bgColor}`}>
                            {icon}
                          </div>

                          {/* Name + Status */}
                          <div className="flex items-center space-x-2">
                            <span
                              title={session.name}
                              className="text-sm font-medium text-gray-900"
                            >
                              {session.name && session.name.length > 40
                                ? `${session.name.substring(0, 40)}...`
                                : session.name || "Untitled Session"}
                            </span>
                            <span
                              className={`px-2 py-1 text-xs font-medium rounded-full ${bgColor} ${textColor}`}
                            >
                              {text}
                            </span>
                          </div>
                        </div>

                        {/* View Link */}
                        <Link
                          to={`/trainings/${session.id}`}
                          className="text-blue-600 text-sm hover:underline"
                        >
                          View
                        </Link>
                      </div>
                    );
                  })}

                  {sessions.length === 0 && (
                    <div className="text-center py-6 text-sm text-gray-500">
                      No active sessions
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Server Stats ---- Hardcoded for now*/}
          {/* <div className="bg-white rounded-xl border-2 border-indigo-300 shadow-sm p-5">
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
          </div> */}
          <div className="bg-white rounded-xl shadow-sm border-2 border-green-300 overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-emerald-50 to-teal-50 px-6 py-4 border-b border-gray-100">
              <div className="flex items-center space-x-3">
                {/* Icon */}
                <div className="bg-emerald-100 p-2 rounded-lg">
                  <svg
                    className="w-5 h-5 text-emerald-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    {/* Server rack outline */}
                    <rect
                      x="3"
                      y="4"
                      width="18"
                      height="16"
                      rx="2"
                      ry="2"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />

                    {/* Server sections */}
                    <line
                      x1="3"
                      y1="10"
                      x2="21"
                      y2="10"
                      strokeWidth={2}
                      strokeLinecap="round"
                    />
                    <line
                      x1="3"
                      y1="16"
                      x2="21"
                      y2="16"
                      strokeWidth={2}
                      strokeLinecap="round"
                    />

                    {/* Health indicator circle (green) */}
                    <circle cx="19" cy="6" r="1.5" fill="currentColor" />
                  </svg>
                </div>

                {/* Title + Subtitle */}
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    Server Health & Status
                  </h2>
                  <p className="text-sm text-gray-500">
                    Real-time system performance metrics
                  </p>
                </div>
              </div>
            </div>

            {/* Body */}
            <div className="p-6">
              {loading.stats ? (
                <div className="flex justify-center py-6">
                  <ArrowPathIcon className="h-5 w-5 text-emerald-500 animate-spin" />
                </div>
              ) : error.stats ? (
                <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-sm text-red-700">
                  {error.stats}
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">
                      Active Training Sessions
                    </span>
                    <span className="text-sm font-semibold text-gray-900">
                      {stats?.active_sessions || 0}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">
                      Pending Client Updates
                    </span>
                    <span className="text-sm font-semibold text-gray-900">
                      {stats?.pending_updates || 0}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">
                      Storage Utilization
                    </span>
                    <span className="text-sm font-semibold text-gray-900">
                      {stats?.storage_used_gb
                        ? `${stats.storage_used_gb}GB / ${stats.storage_total_gb}GB`
                        : "--"}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">
                      Last Model Accuracy
                    </span>
                    <span className="text-sm font-semibold text-gray-900">
                      {stats?.last_accuracy
                        ? `${(stats.last_accuracy * 100).toFixed(1)}%`
                        : "--"}
                    </span>
                  </div>

                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-500">
                      Avg Round Time
                    </span>
                    <span className="text-sm font-semibold text-gray-900">
                      {stats?.avg_round_time
                        ? `${stats.avg_round_time}s`
                        : "--"}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Datasets Section with improved spacing */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Available Datasets */}

          <div className="bg-white rounded-xl shadow-sm border-2 border-indigo-300 overflow-hidden">
            {/* Header - Gradient */}
            <div className="bg-gradient-to-r from-indigo-50 to-blue-50 px-6 py-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                {/* Left section */}
                <div className="flex items-center space-x-3">
                  {/* Icon */}
                  <div className="bg-indigo-100 p-2 rounded-lg">
                    <svg
                      className="w-5 h-5 text-indigo-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 7h10M7 11h10M7 15h6"
                      />
                    </svg>
                  </div>

                  {/* Title + Subtitle */}
                  <div>
                    <div className="flex items-center gap-1">
                      <h2 className="text-lg font-semibold text-gray-900">
                        Recent Datasets
                      </h2>

                      {/* Tooltip */}
                      <div className="relative group">
                        <InformationCircleIcon className="h-5 w-5 text-gray-500 hover:text-gray-700 cursor-help" />

                        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-1 z-30 w-64 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                          View your recently uploaded datasets here. Includes
                          both raw and processed datasets that are ready for
                          federated learning.
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 border-4 border-transparent border-b-gray-800"></div>
                        </div>
                      </div>
                    </div>

                    <p className="text-sm text-gray-500">
                      Showing latest {showDatasets} datasets
                    </p>
                  </div>
                </div>

                {/* View All Button */}
                <Link
                  to="/ManageData"
                  className="inline-flex items-center px-3 py-2 text-sm font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 transition-colors"
                >
                  View All
                  <ArrowRightIcon className="ml-1 h-4 w-4" />
                </Link>
              </div>
            </div>

            {/* Filter Row - Outside Gradient */}
            <div className="px-6 py-4 border-b border-gray-100 flex justify-end items-center gap-2">
              <span className="text-sm font-medium text-gray-600">Filter:</span>
              <div className="inline-flex bg-white border border-gray-200 rounded-lg p-1 shadow-sm">
                <button
                  onClick={() => setShowDatasets("raw")}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                    showDatasets === "raw"
                      ? "bg-blue-100 text-blue-700"
                      : "text-gray-600 hover:text-gray-800"
                  }`}
                >
                  Raw Data
                </button>
                <button
                  onClick={() => setShowDatasets("processed")}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                    showDatasets === "processed"
                      ? "bg-green-100 text-green-700"
                      : "text-gray-600 hover:text-gray-800"
                  }`}
                >
                  Processed
                </button>
              </div>
            </div>

            {/* Body */}
            <div className="p-6 space-y-3">
              {/* {console.log("hello::", datasets.uploads)} */}
              {(showDatasets === "raw" ? datasets.uploads : datasets.processed)
                .slice(0, 4)
                .map((dataset) => (
                  <div
                    key={dataset.filename}
                    className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg transition cursor-pointer"
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
                        <p className="text-sm font-medium text-gray-900 truncate max-w-[200px]">
                          {dataset.filename}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(dataset.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <Link
                      to={`/${showDatasets === "raw" ? "raw" : "processed"}-dataset-overview/${dataset.filename}`}
                      className="text-gray-400 hover:text-gray-600 transition"
                      title="View details"
                    >
                      <EyeIcon className="w-5 h-5" />
                    </Link>
                  </div>
                ))}

              {((showDatasets === "raw" && !datasets.uploads.length) ||
                (showDatasets === "processed" &&
                  !datasets.processed.length)) && (
                <div className="text-center py-6 text-sm text-gray-500">
                  No {showDatasets} datasets found
                </div>
              )}
            </div>
          </div>

          {/* QPD Datasets */}
          {/* <div className="bg-white rounded-xl border-2 border-indigo-300 shadow-sm p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-900">
                Pending QPD
              </h2>
              <button
                className="text-indigo-600 text-xs hover:underline"
                onClick={() => navigate("/assess-data-quality")}
              >
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
          </div> */}
          <div className="bg-white rounded-xl shadow-sm border-2 border-yellow-300 overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-amber-50 to-yellow-50 px-6 py-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {/* Icon */}
                  <div className="bg-amber-100 p-2 rounded-lg">
                    <svg
                      className="w-5 h-5 text-amber-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6M7 4h10a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6a2 2 0 012-2z"
                      />
                    </svg>
                  </div>

                  {/* Title + Subtitle */}
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">
                      Pending QPD
                    </h2>
                    <p className="text-sm text-gray-500">
                      {qpdDatasets.length} dataset
                      {qpdDatasets.length !== 1 ? "s" : ""} awaiting review
                    </p>
                  </div>
                </div>

                {/* View All Button */}
                <button
                  onClick={() => navigate("/assess-data-quality")}
                  className="inline-flex items-center px-3 py-2 text-sm font-medium rounded-md text-amber-700 bg-amber-100 hover:bg-amber-200 transition-colors"
                >
                  View All
                  <svg
                    className="ml-1 w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </button>
              </div>
            </div>

            {/* Body */}
            <div className="p-6">
              {loading.qpd ? (
                <div className="flex justify-center py-6">
                  <ArrowPathIcon className="h-5 w-5 text-amber-500 animate-spin" />
                </div>
              ) : error.qpd ? (
                <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded text-sm text-red-700">
                  {error.qpd}
                </div>
              ) : (
                <div className="space-y-3">
                  {qpdDatasets.slice(0, 4).map((qpd) => (
                    <div
                      key={qpd.id}
                      className="group flex justify-between items-start p-3 hover:bg-gray-50 rounded-lg transition"
                    >
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate max-w-[200px]">
                          {qpd.training_name}
                        </p>

                        <div className="flex flex-wrap items-center mt-1 gap-x-4 gap-y-1">
                          <span className="text-xs text-gray-500">
                            {qpd.num_datapoints} data points
                          </span>

                          <span className="text-xs text-gray-500 truncate max-w-[160px]">
                            Parent: {qpd.parent_filename}
                          </span>

                          <span className="text-xs text-gray-400">
                            {new Date(qpd.transferredAt).toLocaleDateString()}
                          </span>

                          <span className="text-xs text-gray-400">
                            ID: {qpd.federated_session_id}
                          </span>
                        </div>
                      </div>

                      {/* Status Badge */}
                      <div className="self-center">
                        {qpd.approvedAt ? (
                          <span className="text-xs px-2 py-1 rounded-full bg-green-100 text-green-700 font-medium">
                            Approved
                          </span>
                        ) : (
                          <span className="text-xs px-2 py-1 rounded-full bg-yellow-100 text-yellow-700 font-medium">
                            Pending
                          </span>
                        )}
                      </div>
                    </div>
                  ))}

                  {qpdDatasets.length === 0 && (
                    <div className="text-center py-6 text-sm text-gray-500">
                      No pending QPD datasets
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
