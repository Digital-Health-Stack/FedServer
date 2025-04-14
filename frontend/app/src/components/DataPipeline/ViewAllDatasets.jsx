import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  FolderIcon,
  DocumentTextIcon,
  ArrowUpTrayIcon,
  XCircleIcon,
} from "@heroicons/react/24/solid";
import Pagination from "./ViewAllFiles/Pagination";
import FileCard from "./ViewAllFiles/FileCard";

const ViewAllDatasets = () => {
  // Environment variables and navigation setup
  const [selectedFolder, setSelectedFolder] = useState("processed");
  const [datasets, setDatasets] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const PAGE_SIZE = 20; // Number of datasets per page
  const endpoints = {
    raw: {
      fetch: process.env.REACT_APP_RAW_DATASETS_ENDPOINT,
      delete: process.env.REACT_APP_DELETE_RAW_ENDPOINT,
      overview: "/raw-dataset-overview",
    },
    processed: {
      fetch: process.env.REACT_APP_PROCESSED_DATASETS_ENDPOINT,
      delete: process.env.REACT_APP_DELETE_PROCESSED_ENDPOINT,
      overview: "/processed-dataset-overview",
    },
  };

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(endpoints[selectedFolder].fetch, {
        params: { skip: (currentPage - 1) * PAGE_SIZE, limit: PAGE_SIZE },
      });
      setDatasets(response.data);
      setTotalCount(response.headers["x-total-count"] || response.data.length);
    } catch (err) {
      setError("Failed to load datasets");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedFolder, currentPage]);

  const handleDelete = async (datasetId) => {
    if (!window.confirm("Permanently delete this dataset?")) return;
    try {
      await axios.delete(endpoints[selectedFolder].delete, {
        params: { dataset_id: datasetId },
      });
      fetchData();
    } catch (err) {
      setError("Deletion failed");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="mx-auto grid grid-cols-[240px_1fr] gap-8 max-w-9xl">
        {/* Sidebar */}
        <div className="space-y-3">
          {["raw", "processed"].map((folder) => (
            <button
              key={folder}
              onClick={() => setSelectedFolder(folder)}
              className={`w-full flex items-center gap-3 p-3 rounded-xl text-left
                ${
                  selectedFolder === folder
                    ? "bg-blue-50 text-blue-700"
                    : "hover:bg-gray-100"
                }`}
            >
              <FolderIcon className="h-5 w-5" />
              {folder.charAt(0).toUpperCase() + folder.slice(1)} Datasets
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="space-y-6">
          {/* Header */}
          <div className="bg-white p-6 rounded-xl shadow-sm border">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold flex items-center gap-3">
                <DocumentTextIcon className="h-8 w-8 text-blue-500" />
                {selectedFolder === "raw"
                  ? "Raw Datasets"
                  : "Processed Datasets"}
              </h1>
              <a
                href="/preprocessing-docs"
                className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
              >
                <ArrowUpTrayIcon className="h-5 w-5" />
                Processing Guidelines
              </a>
            </div>
          </div>

          {/* Content */}
          {error && (
            <div className="bg-red-50 p-4 rounded-lg flex items-center gap-3">
              <XCircleIcon className="h-5 w-5 text-red-500" />
              <span className="text-red-600">{error}</span>
            </div>
          )}

          {loading ? (
            <div className="animate-pulse space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-20 bg-gray-200 rounded-xl" />
              ))}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {datasets.map((dataset) => (
                  <FileCard
                    key={dataset.dataset_id}
                    dataset={dataset}
                    isRaw={selectedFolder === "raw"}
                    onDelete={handleDelete}
                    onClick={() =>
                      navigate(
                        `${endpoints[selectedFolder].overview}/${dataset.filename}`
                      )
                    }
                    onEditSuccess={fetchData}
                  />
                ))}
              </div>

              <Pagination
                currentPage={currentPage}
                totalCount={totalCount}
                pageSize={PAGE_SIZE}
                onPageChange={setCurrentPage}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ViewAllDatasets;
