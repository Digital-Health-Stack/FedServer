import { useEffect, useState } from "react";
import { toast } from "react-toastify";
import SummaryStats from "../DataPipeline/DataSetVisuals/DatasetDetails/SummaryStats.jsx";
import ColumnDetails from "../DataPipeline/DataSetVisuals/DatasetDetails/ColumnDetails.jsx";
import {
  DocumentTextIcon,
  CircleStackIcon,
  FolderIcon,
  CalendarIcon,
  CheckBadgeIcon,
  LinkIcon,
  ClipboardDocumentIcon,
} from "@heroicons/react/24/outline";
import DataStats from "./DataStats";
import Spinner from "./Spinner";
import {
  getTransferredDataOverview,
  approveDataTransfer,
} from "../../services/privateService";
import Loader from "./Loader";
import { motion } from "framer-motion";

const DataTransferDetails = ({ transferId }) => {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [approving, setApproving] = useState(false);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard!", {
      position: "bottom-center",
      autoClose: 2000,
    });
  };

  const handleApprove = async () => {
    try {
      setApproving(true);
      await approveDataTransfer(transferId);
      setDetails((prev) => ({ ...prev, approvedAt: new Date().toISOString() }));
      toast.success("Data approved successfully!", {
        position: "bottom-center",
        autoClose: 3000,
      });
    } catch (err) {
      toast.error(`Approval failed: ${err.message}`, {
        position: "bottom-center",
        autoClose: 4000,
      });
    } finally {
      setApproving(false);
    }
  };

  useEffect(() => {
    const fetchDetails = async () => {
      try {
        const response = await getTransferredDataOverview(transferId);
        setDetails(response.data);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [transferId]);

  if (loading) return <Spinner />;
  if (!details) return <div className="text-gray-500">No details found</div>;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full overflow-y-auto p-8"
    >
      <div className="max-w-5xl mx-auto bg-white rounded-2xl shadow-lg p-8">
        {/* Header Section */}
        <div className="flex items-start justify-between mb-8">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-100 rounded-xl">
              <DocumentTextIcon className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                {details.training_name}
              </h1>
              <p className="text-gray-500 mt-1">Transfer ID: {details.id}</p>
            </div>
          </div>

          <button
            onClick={handleApprove}
            disabled={details.approvedAt || approving}
            className={`px-6 py-2 rounded-lg font-medium transition-all ${
              details.approvedAt
                ? "bg-green-100 text-green-700 cursor-default"
                : "bg-blue-600 hover:bg-blue-700 text-white"
            } ${approving && "opacity-50"}`}
          >
            {approving ? (
              <Loader className="w-5 h-5" />
            ) : details.approvedAt ? (
              "Approved ✓"
            ) : (
              "Approve Transfer"
            )}
          </button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <DetailItem
            icon={CircleStackIcon}
            label="Data Points"
            value={details.num_datapoints.toLocaleString()}
            className="bg-blue-50"
          />
          <DetailItem
            icon={FolderIcon}
            label="Parent File"
            value={details.parent_filename.slice(0, 15) + "..."}
            className="bg-purple-50"
          />
          <DetailItem
            icon={CalendarIcon}
            label="Transferred At"
            value={new Date(details.transferredAt).toLocaleDateString()}
            className="bg-amber-50"
          />
          <DetailItem
            icon={CheckBadgeIcon}
            label="Status"
            value={details.approvedAt ? "Approved" : "Pending"}
            statusColor={
              details.approvedAt ? "text-green-600" : "text-amber-600"
            }
            className="bg-emerald-50"
          />
        </div>

        {/* Data Path Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium flex items-center gap-2 text-gray-700">
              <LinkIcon className="w-5 h-5 text-gray-500" />
              Data Path
            </h3>
            <button
              onClick={() => copyToClipboard(details.data_path)}
              className="text-blue-600 hover:text-blue-700 flex items-center gap-1"
            >
              <ClipboardDocumentIcon className="w-4 h-4" />
              <span className="text-sm">Copy</span>
            </button>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg break-all font-mono text-sm border border-gray-200">
            {details.data_path}
          </div>
        </div>

        {/* Data Statistics */}
        <div className="border-t border-gray-100 pt-8">
          <h3 className="text-xl font-semibold mb-6 text-gray-900">
            Dataset Statistics
          </h3>
          {/* <DataStats stats={details.datastats} /> */}
          <SummaryStats
            filename={details.parent_filename}
            numRows={details.datastats.numRows}
            numCols={details.datastats.numColumns}
          />
          <ColumnDetails columnStats={details.datastats.columnStats} />
        </div>
      </div>
    </motion.div>
  );
};

const DetailItem = ({ icon: Icon, label, value, statusColor, className }) => (
  <div className={`p-4 rounded-xl ${className}`}>
    <div className="flex items-center gap-3">
      <div className="p-2 bg-white rounded-lg shadow-sm border border-gray-100">
        <Icon className="w-6 h-6 text-gray-600" />
      </div>
      <div>
        <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          {label}
        </div>
        <div
          className={`text-lg font-semibold ${statusColor || "text-gray-900"}`}
        >
          {value}
        </div>
      </div>
    </div>
  </div>
);

export default DataTransferDetails;
