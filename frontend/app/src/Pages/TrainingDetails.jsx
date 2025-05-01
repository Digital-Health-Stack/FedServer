import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { useForm } from "react-hook-form";
import { useAuth } from "../contexts/AuthContext";
import { getFederatedSession } from "../services/federatedService";
import axios from "axios";
import FederatedSessionLogs from "../components/Training/FederatedSessionLogs";
import SessionInfo from "../components/Training/SessionInfo";
import DatasetInfo from "../components/Training/DatasetInfo";
import ModelConfig from "../components/Training/ModelConfig";
import {
  ArrowPathIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  CogIcon,
  ServerStackIcon,
  InformationCircleIcon,
  BoltIcon,
} from "@heroicons/react/24/outline";
import Result from "../components/Training/Result";

const clientPriceResponseEndpoint =
  process.env.REACT_APP_SUBMIT_CLIENT_PRICE_RESPONSE_URL;

const statusConfig = {
  1: {
    text: "Pending Start",
    color: "bg-blue-100 text-blue-800",
    icon: <ClockIcon className="h-5 w-5" />,
  },
  2: {
    text: "Initializing",
    color: "bg-blue-100 text-blue-800",
    icon: <CogIcon className="h-5 w-5" />,
  },
  3: {
    text: "Awaiting Clients",
    color: "bg-blue-100 text-blue-800",
    icon: <InformationCircleIcon className="h-5 w-5" />,
  },
  4: {
    text: "Training Active",
    color: "bg-yellow-100 text-yellow-800",
    icon: <BoltIcon className="h-5 w-5" />,
  },
  5: {
    text: "Completed",
    color: "bg-green-100 text-green-800",
    icon: <CheckCircleIcon className="h-5 w-5" />,
  },
  "-1": {
    text: "Failed",
    color: "bg-red-100 text-red-800",
    icon: <ExclamationTriangleIcon className="h-5 w-5" />,
  },
};

export default function TrainingDetails() {
  const { sessionId } = useParams();
  const { api } = useAuth();
  const [federatedSessionData, setFederatedSessionData] = useState({});
  const [currentSection, setCurrentSection] = useState("session-info");
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchFederatedSessionData = async () => {
    try {
      setIsRefreshing(true);
      console.log("Checkpoint 1: ", sessionId)
      const response = await getFederatedSession(api, sessionId);
      setFederatedSessionData(response.data);
      console.log("Checkpoint 1: ", response.data);
    } catch (error) {
      console.error("Error fetching session data:", error);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchFederatedSessionData();
  }, [sessionId]);


  const sections = [
    {
      id: "session-info",
      label: "Session Information",
      icon: <InformationCircleIcon className="h-5 w-5" />,
    },
    {
      id: "dataset-info",
      label: "Dataset",
      icon: <ServerStackIcon className="h-5 w-5" />,
    },
    {
      id: "model-config",
      label: "Model Config",
      icon: <CogIcon className="h-5 w-5" />,
    },
    {
      id: "session-logs",
      label: "Logs",
      icon: <ChartBarIcon className="h-5 w-5" />,
    },
          {
            id: "results",
            label: "Results",
            icon: <ChartBarIcon className="h-5 w-5" />,
          },
  ];

  const renderStatusBadge = () => {
    const status = federatedSessionData?.training_status;
    const config = statusConfig[status] || statusConfig["-1"];

    return (
      <span
        className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${config.color}`}
      >
        {config.icon}
        <span className="ml-2">{config.text}</span>
      </span>
    );
  };

  if (!federatedSessionData) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center p-8 bg-white rounded-lg shadow-sm max-w-md">
          <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-4 text-lg font-medium text-gray-900">
            No session data
          </h3>
          <p className="mt-2 text-sm text-gray-600">
            The requested training session could not be found or loaded.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex bg-gray-50 min-h-screen">
      {/* Sidebar Navigation */}
      <div className="w-64 bg-white shadow-sm p-4 border-r border-gray-200">
        <div className="flex items-center justify-between mb-6 p-2">
          <h3 className="text-lg font-semibold text-gray-800">
            Session Navigation
          </h3>
          <button
            onClick={fetchFederatedSessionData}
            disabled={isRefreshing}
            className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-50"
          >
            <ArrowPathIcon
              className={`h-5 w-5 ${isRefreshing ? "animate-spin" : ""}`}
            />
          </button>
        </div>

        <div className="mb-4 p-2">{renderStatusBadge()}</div>

        <nav className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setCurrentSection(section.id)}
              className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition ${
                currentSection === section.id
                  ? "bg-blue-100 text-blue-700"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <span className="mr-3">{section.icon}</span>
              {section.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          {currentSection === "session-info" && (
            <SessionInfo data={federatedSessionData} />
          )}
          {currentSection === "session-logs" && (
            <FederatedSessionLogs sessionId={sessionId} />
          )}
          {currentSection === "dataset-info" && (
            <DatasetInfo data={federatedSessionData.federated_info} />
          )}
          {currentSection === "model-config" && (
            <ModelConfig data={federatedSessionData.federated_info} />
          )}
          {currentSection === "results"  && (
            <Result sessionId={sessionId} />  
          )}
        </div>
      </div>
    </div>
  );
}
