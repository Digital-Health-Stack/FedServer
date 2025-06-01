import {
  InformationCircleIcon,
  BuildingOfficeIcon,
  CpuChipIcon,
  ScaleIcon,
  ArrowPathIcon,
  UserGroupIcon,
  CurrencyDollarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CogIcon,
  XCircleIcon,
  DocumentPlusIcon,
  Cog6ToothIcon,
  CheckBadgeIcon
} from '@heroicons/react/24/outline';
import React, { useEffect } from "react";

const SessionInfo = ({ data }) => {
  // Training Status Configuration
  const TRAINING_STATUS = {
    0: { 
      label: "Session Created", 
      variant: "neutral", 
      icon: <DocumentPlusIcon className="h-5 w-5" />,
    },
    1: { 
      label: "Price Negotiation", 
      variant: "warning", 
      icon: <CurrencyDollarIcon className="h-5 w-5" />,
    },
    2: { 
      label: "Client Recruitment", 
      variant: "info", 
      icon: <UserGroupIcon className="h-5 w-5" />,
    },
    3: { 
      label: "Model Initialization", 
      variant: "primary", 
      icon: <Cog6ToothIcon className="h-5 w-5" />,
    },
    4: { 
      label: "Training Active", 
      variant: "primary", 
      icon: <ArrowPathIcon className="h-5 w-5 animate-spin" />,
    },
    5: { 
      label: "Completed", 
      variant: "success", 
      icon: <CheckBadgeIcon className="h-5 w-5" />,
      description: "Training successfully completed"
    },
    [-1]: { 
      label: "Failed", 
      variant: "danger", 
      icon: <ExclamationTriangleIcon className="h-5 w-5" />,
    }
  };

  // Client Status Configuration
  const CLIENT_STATUS = {
    0: { label: "Accepted", variant: "success", icon: <CheckCircleIcon className="h-5 w-5" /> },
    1: { label: "Initialized", variant: "primary", icon: <CogIcon className="h-5 w-5" /> },
    [-1]: { label: "Rejected", variant: "danger", icon: <XCircleIcon className="h-5 w-5" /> }
  };

  useEffect(() => {
    console.log("Training Status Value:", data);
  }, [data]);

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Session Information Header */}
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900 flex items-center">
          <InformationCircleIcon className="h-5 w-5 text-indigo-600 mr-2" />
          Session Details
        </h3>
      </div>

      {/* Session Details Content */}
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          <InfoItem 
            label="Organisation Name" 
            value={data?.federated_info?.organisation_name || "N/A"} 
            icon={<BuildingOfficeIcon className="h-5 w-5 text-gray-400" />}
          />
          <InfoItem 
            label="Model Name" 
            value={data?.federated_info?.model_name || "N/A"} 
            icon={<CpuChipIcon className="h-5 w-5 text-gray-400" />}
          />
          <InfoItem 
            label="Expected Value" 
            value={data?.federated_info?.expected_results?.std_mean} 
            icon={<ScaleIcon className="h-5 w-5 text-gray-400" />}
          />
          <InfoItem 
            label="Expected Deviation" 
            value={data?.federated_info?.expected_results?.std_deviation} 
            icon={<ArrowPathIcon className="h-5 w-5 text-gray-400" />}
          />
          <StatusItem
            label="Training Status"
            statusConfig={TRAINING_STATUS[data?.training_status] || TRAINING_STATUS[0]}
          />
          <StatusItem
            label="Client Status"
            statusConfig={CLIENT_STATUS[data?.client_status] || CLIENT_STATUS[-1]}
          />
        </div>
      </div>
    </div>
  );
};

// Supporting components
const InfoItem = ({ label, value, icon }) => (
  <div className="flex items-start space-x-3">
    <div className="flex-shrink-0 mt-0.5">
      {icon}
    </div>
    <div>
      <p className="text-sm font-medium text-gray-500">{label}</p>
      <p className="text-base font-semibold text-gray-900">{value}</p>
    </div>
  </div>
);

const StatusItem = ({ label, statusConfig }) => {
  const variantClasses = {
    primary: "bg-blue-50 text-blue-800",
    success: "bg-green-50 text-green-800",
    warning: "bg-yellow-50 text-yellow-800",
    danger: "bg-red-50 text-red-800",
    info: "bg-cyan-50 text-cyan-800",
    neutral: "bg-gray-50 text-gray-800"
  };

  return (
    <div className="flex items-start space-x-3">
      <div className="flex-shrink-0 mt-0.5">
        {React.cloneElement(statusConfig.icon, { 
          className: `h-5 w-5 ${variantClasses[statusConfig.variant].replace('bg-', 'text-').split(' ')[1]}` 
        })}
      </div>
      <div>
        <p className="text-sm font-medium text-gray-500">{label}</p>
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variantClasses[statusConfig.variant]}`}>
          {statusConfig.label}
        </span>
      </div>
    </div>
  );
};

export default SessionInfo;
