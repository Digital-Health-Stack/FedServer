import React, { useEffect, useState, useRef } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { getLogsSession } from "../../services/federatedService";
import { ArrowPathIcon, ExclamationTriangleIcon, InformationCircleIcon } from "@heroicons/react/24/solid";

const FederatedSessionLogs = ({ sessionId }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const { api } = useAuth();
  const logsEndRef = useRef(null);

  const fetchLogsSessionData = async () => {
    if (!sessionId) {
      setLogs([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError("");
    try {
      const response = await getLogsSession(api, sessionId);
      // Reverse the logs to show most recent first
      setLogs(response.data.reverse());
    } catch (err) {
      console.error("Error fetching logs:", err);
      setError("Failed to fetch logs. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogsSessionData();
  }, [sessionId]);

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    if (logs.length > 0 && !loading) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, loading]);

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

  return (
    <div className="w-full">
      <div className="bg-white shadow rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 flex items-center">
            Training Logs
            {loading && (
              <ArrowPathIcon className="h-5 w-5 ml-2 text-blue-500 animate-spin" />
            )}
          </h3>
        </div>

        <div className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    Timestamp
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    Message
                  </th>
                </tr>
              </thead>
            </table>
            <div className="max-h-[500px] overflow-y-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <tbody className="bg-white divide-y divide-gray-200">
                  {loading ? (
                    <tr>
                      <td colSpan="2" className="px-6 py-4 text-center">
                        <div className="flex justify-center items-center text-gray-500">
                          <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
                          Loading logs...
                        </div>
                      </td>
                    </tr>
                  ) : error ? (
                    <tr>
                      <td colSpan="2" className="px-6 py-4 text-center">
                        <div className="flex justify-center items-center text-red-500">
                          <ExclamationTriangleIcon className="h-5 w-5 mr-2" />
                          {error}
                        </div>
                      </td>
                    </tr>
                  ) : logs.length === 0 ? (
                    <tr>
                      <td colSpan="2" className="px-6 py-4 text-center">
                        <div className="flex justify-center items-center text-gray-500">
                          <InformationCircleIcon className="h-5 w-5 mr-2" />
                          {sessionId ? "No logs available" : "Select a session to view logs"}
                        </div>
                      </td>
                    </tr>
                  ) : (
                    logs.map((log) => (
                      <tr key={log.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatTimestamp(log.created_at)}
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          {log.message}
                        </td>
                      </tr>
                    ))
                  )}
                  <tr ref={logsEndRef} />
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="px-6 py-3 bg-gray-50 text-right text-xs text-gray-500 border-t border-gray-200">
          {sessionId && `Showing logs for Session ID #${sessionId}`}
        </div>
      </div>
    </div>
  );
};

export default FederatedSessionLogs;