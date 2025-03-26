import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  TrashIcon,
  Cog6ToothIcon,
  DocumentTextIcon,
  FolderIcon,
  XCircleIcon,
  ArrowUpTrayIcon,
} from "@heroicons/react/24/solid";

const REACT_APP_LIST_ALL_DATASETS_URL =
  process.env.REACT_APP_LIST_ALL_DATASETS_URL;
const REACT_APP_DELETE_DATAFILE_URL = process.env.REACT_APP_DELETE_DATAFILE_URL;
const REACT_APP_DATASET_OVERVIEW_PATH = "/dataset-overview";
const REACT_APP_PROCESSING_GUIDELINES_PATH = "/preprocessing-docs";

const ViewAllDatasets = () => {
  const [contents, setContents] = useState({});
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await axios.get(REACT_APP_LIST_ALL_DATASETS_URL);
        setContents(response.data.contents || {});
      } catch (err) {
        setError("Failed to load datasets. Please refresh the page.");
        console.error(err);
      }
    };
    fetchFiles();
  }, []);

  const handleDelete = async (file, directory) => {
    if (!window.confirm(`Permanently delete ${file}?`)) return;

    try {
      await axios.delete(REACT_APP_DELETE_DATAFILE_URL, {
        params: { directory, fileName: file },
      });

      setContents((prev) => ({
        ...prev,
        [directory]: prev[directory].filter((f) => f !== file),
      }));
    } catch (err) {
      setError("Deletion failed. Ensure file isn't being processed.");
      console.error(err);
    }
  };

  const DirectorySection = ({ name, files }) => (
    <div className="mb-8 bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="flex items-center gap-3 mb-4">
        <FolderIcon className="h-6 w-6 text-blue-500" />
        <h2 className="text-xl font-semibold text-gray-800">
          {name.replace(/^\w/, (c) => c.toUpperCase())}
        </h2>
        <span className="text-sm text-gray-500 ml-2">
          ({files.length} file{files.length !== 1 ? "s" : ""})
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {files.map((file) => (
          <FileCard
            key={file}
            file={file}
            directory={name}
            onDelete={handleDelete}
            onClick={() =>
              navigate(
                `${REACT_APP_DATASET_OVERVIEW_PATH}/${encodeURIComponent(
                  name
                )}/${encodeURIComponent(file)}`
              )
            }
          />
        ))}
      </div>
    </div>
  );

  const FileCard = ({ file, directory, onDelete, onClick }) => {
    const isProcessing = file.endsWith("__PROCESSING__");
    const displayName = file.replace(/__PROCESSING__$/, "");

    return (
      <div
        className={`group relative p-4 rounded-lg border transition-all
          ${
            isProcessing
              ? "border-yellow-200 bg-yellow-50 cursor-wait"
              : "border-gray-200 hover:border-blue-200 hover:bg-blue-50 cursor-pointer"
          }
        `}
        onClick={!isProcessing ? onClick : undefined}
      >
        <div className="flex justify-between items-start">
          <div className="min-w-0">
            <p
              className="text-gray-700 font-medium truncate"
              title={displayName}
            >
              {displayName}
            </p>
            {isProcessing && (
              <div className="text-sm text-yellow-600 mt-2 flex items-center">
                <Cog6ToothIcon className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </div>
            )}
          </div>

          {!isProcessing && (
            <TrashIcon
              className="h-5 w-5 text-red-400 hover:text-red-600 shrink-0 ml-2"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(file, directory);
              }}
            />
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header Section */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <DocumentTextIcon className="h-8 w-8 text-blue-500" />
              Dataset Manager
            </h1>

            <a
              href={REACT_APP_PROCESSING_GUIDELINES_PATH}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
            >
              <ArrowUpTrayIcon className="h-5 w-5" />
              View Processing Guidelines
            </a>
          </div>

          {error && (
            <div className="bg-red-50 p-4 rounded-lg flex items-center gap-3">
              <XCircleIcon className="h-5 w-5 text-red-500" />
              <span className="text-red-600">{error}</span>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-600 hover:text-red-800"
              >
                Dismiss
              </button>
            </div>
          )}
        </div>

        {/* Files Display */}
        <div className="space-y-8">
          {Object.keys(contents).length > 0 ? (
            Object.entries(contents)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([directory, files]) => (
                <DirectorySection
                  key={directory}
                  name={directory}
                  files={files.sort((a, b) => a.localeCompare(b))}
                />
              ))
          ) : (
            <div className="text-center py-12 bg-white rounded-xl border border-gray-200">
              <p className="text-gray-500 mb-4">No datasets found</p>
              <button
                onClick={() => navigate("/ManageData")}
                className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors"
              >
                Manage Data Page
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ViewAllDatasets;
