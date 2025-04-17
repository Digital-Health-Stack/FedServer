// AssessDataQuality.jsx (Main Page)
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  DocumentTextIcon,
  ChevronDoubleRightIcon,
  ChevronDoubleLeftIcon,
} from "@heroicons/react/24/solid";
import DataTransferList from "../components/AssessDataTransfer/DataTransferList";
import DataTransferDetails from "../components/AssessDataTransfer/DataTransferDetails";
import Spinner from "../components/AssessDataTransfer/Spinner";
import { listTransferredData } from "../services/privateService";

const AssessDataQuality = () => {
  const [transfers, setTransfers] = useState([]);
  const [selectedTransfer, setSelectedTransfer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({ skip: 0, limit: 20 });
  const [hasMore, setHasMore] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const loadMore = async () => {
    try {
      const newSkip = pagination.skip + pagination.limit;
      const response = await listTransferredData({
        skip: newSkip,
        limit: pagination.limit,
      });

      if (response.data.length === 0) {
        setHasMore(false);
        return;
      }

      setTransfers((prev) => [...prev, ...response.data]);
      setPagination((prev) => ({ ...prev, skip: newSkip }));
    } catch (err) {
      console.error("Error loading more:", err);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await listTransferredData(pagination);
        setTransfers(response.data);

        // Check URL hash first
        const hashId = window.location.hash.replace("#", "");
        const initialSelection = hashId
          ? response.data.find((t) => t.id === hashId)?.id
          : response.data[0]?.id;

        setSelectedTransfer(initialSelection || null);
        setHasMore(response.data.length >= pagination.limit);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleSelectTransfer = (transferId) => {
    setSelectedTransfer(transferId);
    window.location.hash = `#${transferId}`;
  };

  if (loading) return <Spinner />;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!transfers.length)
    return <div className="p-4 text-gray-500">No data transfers found</div>;

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Collapse Button */}
      <button
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        className="absolute bottom-4 left-2 z-20 p-2 rounded-full bg-white shadow-lg hover:bg-gray-100"
      >
        {isSidebarOpen ? (
          <ChevronDoubleLeftIcon className="w-5 h-5 text-blue-600" />
        ) : (
          <ChevronDoubleRightIcon className="w-5 h-5 text-blue-600" />
        )}
      </button>

      {/* Left Panel */}
      <AnimatePresence initial={false}>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "25%" }}
            exit={{ opacity: 0, width: 0 }}
            className="flex-shrink-0 h-full bg-white border-r border-gray-200"
          >
            <div className="p-6 border-b border-gray-200">
              <h1 className="text-2xl font-bold flex items-center gap-3 text-blue-800">
                <DocumentTextIcon className="w-8 h-8 text-blue-600" />
                Data Transfers
              </h1>
            </div>
            <DataTransferList
              transfers={transfers}
              selectedId={selectedTransfer}
              onSelect={handleSelectTransfer}
              loadMore={loadMore}
              hasMore={hasMore}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Right Panel */}
      <motion.div
        className={`flex-grow h-full transition-all duration-300 ${
          isSidebarOpen ? "w-3/4" : "w-full"
        }`}
      >
        <div className="h-full overflow-auto">
          {selectedTransfer ? (
            <DataTransferDetails
              transferId={selectedTransfer}
              key={selectedTransfer}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-2xl text-gray-400">
              Select a transfer to begin analysis
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
};
export default AssessDataQuality;
