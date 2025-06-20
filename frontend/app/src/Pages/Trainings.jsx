import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { getAllSessions } from "../services/federatedService";
import {
  ArrowPathIcon,
  ExclamationTriangleIcon,
  CubeIcon,
  ArrowRightIcon,
  ClockIcon,
  ChartBarIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "@heroicons/react/24/outline";

export default function Trainings() {
  const navigate = useNavigate();
  const { api } = useAuth();
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({
    page: 1,
    perPage: 6, // Match your grid layout (3 cols × 2 rows)
    total: 0,
    totalPages: 1,
  });

  const openDetails = (sessionId) => {
    navigate(`/trainings/${sessionId}`);
  };

  const fetchSessions = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await getAllSessions(
        api,
        pagination.page,
        pagination.perPage,
      );
      console.log("Checkpoint 1: ", response.data, typeof response.data);
      setSessions(response.data.data || []);
      setPagination((prev) => ({
        ...prev,
        total: response.data.total,
        totalPages: Math.ceil(response.data.total / pagination.perPage), // Ensure correct page count
      }));
    } catch (err) {
      console.error("Error fetching sessions:", err);
      setError("Failed to load training sessions");
    } finally {
      setIsLoading(false);
    }
  };
  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= pagination.totalPages) {
      setPagination((prev) => ({ ...prev, page: newPage }));
    }
  };

  useEffect(() => {
    fetchSessions();
  }, [pagination.page]);

  const getStatusColor = (status) => {
    switch (status) {
      case 0:
        return "bg-gray-100 text-gray-800"; // Session Created
      case 1:
        return "bg-yellow-100 text-yellow-800"; // Price Negotiation
      case 2:
        return "bg-blue-100 text-blue-800"; // Client Recruitment
      case 3:
        return "bg-indigo-100 text-indigo-800"; // Model Initialization
      case 4:
        return "bg-purple-100 text-purple-800"; // Training Active
      case 5:
        return "bg-green-100 text-green-800"; // Completed
      case -1:
        return "bg-red-100 text-red-800"; // Failed
      default:
        return "bg-gray-100 text-gray-800"; // Unknown
    }
  };

  const TrainingStatuses = {
    0: "Session Created",
    1: "Price Negotiation",
    2: "Client Recruitment",
    3: "Model Initialization",
    4: "Training Active",
    5: "Completed",
    [-1]: "Failed",
  };
  const formatTimestamp = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString("en-IN", {
        dateStyle: "medium",
        timeStyle: "short",
      });
    } catch (e) {
      return timestamp;
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[300px]">
        <ArrowPathIcon className="h-12 w-12 text-blue-500 animate-spin" />
        <p className="mt-4 text-lg font-medium text-gray-700">
          Loading training sessions...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
            <h3 className="text-lg font-medium text-red-800">
              Error loading data
            </h3>
          </div>
          <p className="mt-2 text-sm text-red-700">{error}</p>
          <button
            onClick={fetchSessions}
            className="mt-3 inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center">
            <CubeIcon className="h-6 w-6 text-blue-600 mr-2" />
            Training Sessions
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Monitor active and completed federated learning sessions
          </p>
        </div>
        <button
          onClick={fetchSessions}
          className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <ArrowPathIcon className="h-4 w-4 mr-2" />
          Refresh
        </button>
      </div>

      {sessions.length === 0 ? (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500 mr-2" />
            <h3 className="text-lg font-medium text-yellow-800">
              No training sessions
            </h3>
          </div>
          <p className="mt-2 text-sm text-yellow-700">
            There are currently no training sessions to display.
          </p>
        </div>
      ) : (
        <>
          <div className="mb-4 flex justify-between items-center">
            <p className="text-sm text-gray-500">
              Showing{" "}
              <span className="font-medium">
                {(pagination.page - 1) * pagination.perPage + 1}
              </span>{" "}
              to{" "}
              <span className="font-medium">
                {Math.min(
                  pagination.page * pagination.perPage,
                  pagination.total,
                )}
              </span>{" "}
              of <span className="font-medium">{pagination.total}</span>{" "}
              training sessions
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {sessions.map((session) => (
              <div
                key={session.id}
                className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => openDetails(session.id)}
              >
                <div className="p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-medium text-gray-900 truncate">
                      {session.name || "Untitled Session"}
                    </h3>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                        session.training_status,
                      )}`}
                    >
                      {TrainingStatuses[session.training_status] || "Unknown"}
                    </span>
                  </div>

                  <div className="flex items-center text-sm text-gray-500 mb-2">
                    <ClockIcon className="flex-shrink-0 h-4 w-4 mr-1.5" />
                    <span>Session ID: {session.id}</span>
                  </div>

                  <div className="flex items-center text-sm text-gray-500">
                    <ChartBarIcon className="flex-shrink-0 h-4 w-4 mr-1.5" />
                    <span>Created: {formatTimestamp(session.created_at)}</span>
                  </div>
                </div>

                <div className="bg-gray-50 px-5 py-3 flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-500">
                    View details
                  </span>
                  <ArrowRightIcon className="h-4 w-4 text-gray-400" />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 flex items-center justify-between border-t border-gray-200 px-4 py-3 sm:px-6">
            <div className="flex flex-1 justify-between sm:hidden">
              <button
                onClick={() => handlePageChange(pagination.page - 1)}
                disabled={pagination.page === 1}
                className="relative inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Previous
              </button>
              <button
                onClick={() => handlePageChange(pagination.page + 1)}
                disabled={pagination.page === pagination.totalPages}
                className="relative ml-3 inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Showing{" "}
                  <span className="font-medium">
                    {(pagination.page - 1) * pagination.perPage + 1}
                  </span>{" "}
                  to{" "}
                  <span className="font-medium">
                    {Math.min(
                      pagination.page * pagination.perPage,
                      pagination.total,
                    )}
                  </span>{" "}
                  of <span className="font-medium">{pagination.total}</span>{" "}
                  results
                </p>
              </div>
              <div>
                <nav
                  className="isolate inline-flex -space-x-px rounded-md shadow-sm"
                  aria-label="Pagination"
                >
                  <button
                    onClick={() => handlePageChange(pagination.page - 1)}
                    disabled={pagination.page === 1}
                    className="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0"
                  >
                    <span className="sr-only">Previous</span>
                    <ChevronLeftIcon className="h-5 w-5" aria-hidden="true" />
                  </button>
                  {Array.from(
                    { length: Math.min(5, pagination.totalPages) },
                    (_, i) => {
                      let pageNum;
                      if (pagination.totalPages <= 5) {
                        pageNum = i + 1;
                      } else if (pagination.page <= 3) {
                        pageNum = i + 1;
                      } else if (pagination.page >= pagination.totalPages - 2) {
                        pageNum = pagination.totalPages - 4 + i;
                      } else {
                        pageNum = pagination.page - 2 + i;
                      }

                      return (
                        <button
                          key={pageNum}
                          onClick={() => handlePageChange(pageNum)}
                          className={`relative inline-flex items-center px-4 py-2 text-sm font-semibold ${
                            pagination.page === pageNum
                              ? "bg-blue-600 text-white focus:z-20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
                              : "text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0"
                          }`}
                        >
                          {pageNum}
                        </button>
                      );
                    },
                  )}
                  <button
                    onClick={() => handlePageChange(pagination.page + 1)}
                    disabled={pagination.page === pagination.totalPages}
                    className="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0"
                  >
                    <span className="sr-only">Next</span>
                    <ChevronRightIcon className="h-5 w-5" aria-hidden="true" />
                  </button>
                </nav>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
