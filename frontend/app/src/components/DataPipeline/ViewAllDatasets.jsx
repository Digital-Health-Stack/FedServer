import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import {
  FolderIcon,
  DocumentTextIcon,
  ArrowUpTrayIcon,
  XCircleIcon,
} from "@heroicons/react/24/solid";
import { FilePlus } from "lucide-react";
import Pagination from "./ViewAllFiles/Pagination";
import FileCard from "./ViewAllFiles/FileCard";
import AddDataset from "./ViewAllDatasetsHelper/AddDataset";

const ViewAllDatasets = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedFolder, setSelectedFolder] = useState("add");
  const [datasets, setDatasets] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    // Support legacy hashes: raw/processed → datasets
    const normalizedHash =
      hash === "raw" || hash === "processed" ? "datasets" : hash;
    if (["add", "datasets"].includes(normalizedHash)) {
      setSelectedFolder(normalizedHash);
    } else {
      setSelectedFolder("add");
      navigate(`${location.pathname}#add`);
    }
  }, [location.hash, location.pathname]);

  const handleTabClick = (folder) => {
    navigate(`${location.pathname}#${folder}`);
    setSelectedFolder(folder);
  };

  const PAGE_SIZE = 20;

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(
        `${process.env.REACT_APP_SERVER_BASE_URL}/list-all-datasets`,
        {
          params: { skip: (currentPage - 1) * PAGE_SIZE, limit: PAGE_SIZE },
        }
      );
      const data = response.data;
      if (data.datasets) {
        setDatasets(data.datasets);
        setTotalCount(data.total || data.datasets.length);
      } else if (Array.isArray(data)) {
        setDatasets(data);
        setTotalCount(data.length);
      } else {
        setDatasets([]);
        setTotalCount(0);
      }
    } catch (err) {
      setError("Failed to load datasets");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedFolder === "datasets") {
      fetchData();
    }
  }, [selectedFolder, currentPage]);

  const handleDelete = async (datasetId, isRaw) => {
    if (!window.confirm("Permanently delete this dataset?")) return;
    try {
      const deleteUrl = isRaw
        ? `${process.env.REACT_APP_SERVER_BASE_URL}/delete-raw-dataset-file`
        : `${process.env.REACT_APP_SERVER_BASE_URL}/delete-dataset-file`;
      await axios.delete(deleteUrl, {
        params: { dataset_id: datasetId },
      });
      fetchData();
    } catch (err) {
      setError("Deletion failed");
    }
  };

  const getOverviewPath = (dataset) => {
    if (dataset.source === "processed") {
      return `/processed-dataset-overview/${dataset.filename}`;
    }
    return `/raw-dataset-overview/${dataset.filename}`;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="mx-auto grid grid-cols-[240px_1fr] gap-8 max-w-9xl">
        {/* Sidebar */}
        <div className="space-y-3">
          <button
            onClick={() => handleTabClick("add")}
            className={`w-full flex items-center gap-3 p-3 rounded-xl text-left
              ${
                selectedFolder === "add"
                  ? "bg-blue-50 text-blue-700"
                  : "hover:bg-gray-100"
              }`}
          >
            <FilePlus className="h-5 w-5" />
            Add New Dataset
          </button>
          <button
            onClick={() => handleTabClick("datasets")}
            className={`w-full flex items-center gap-3 p-3 rounded-xl text-left
              ${
                selectedFolder === "datasets"
                  ? "bg-blue-50 text-blue-700"
                  : "hover:bg-gray-100"
              }`}
          >
            <FolderIcon className="h-5 w-5" />
            Datasets
          </button>
        </div>

        {selectedFolder === "add" && <AddDataset />}
        {/* Main Content */}
        {selectedFolder === "datasets" && (
          <div className="space-y-6">
            {/* Header */}
            <div className="bg-white p-6 rounded-xl shadow-sm border">
              <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold flex items-center gap-3">
                  <DocumentTextIcon className="h-8 w-8 text-blue-500" />
                  Datasets
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
                {datasets.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    <DocumentTextIcon className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                    <p className="text-lg font-medium">No datasets found</p>
                    <p className="text-sm mt-2">
                      Upload files using the "Add New Dataset" tab
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {datasets.map((dataset) => (
                        <FileCard
                          key={`${dataset.source}-${dataset.dataset_id || dataset.filename}`}
                          dataset={dataset}
                          isRaw={dataset.source === "raw"}
                          onDelete={(id) =>
                            handleDelete(id, dataset.source === "raw")
                          }
                          onClick={() => navigate(getOverviewPath(dataset))}
                          onEditSuccess={fetchData}
                        />
                      ))}
                    </div>

                    {totalCount > PAGE_SIZE && (
                      <Pagination
                        currentPage={currentPage}
                        totalCount={totalCount}
                        pageSize={PAGE_SIZE}
                        onPageChange={setCurrentPage}
                      />
                    )}
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ViewAllDatasets;
