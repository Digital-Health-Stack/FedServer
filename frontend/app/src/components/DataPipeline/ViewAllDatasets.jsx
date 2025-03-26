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
  ChevronLeftIcon,
  ChevronRightIcon,
} from "@heroicons/react/24/solid";

const PAGE_SIZE = 20;

const ViewAllDatasets = () => {
  // Environment variables
  const RAW_DATASETS_ENDPOINT = process.env.REACT_APP_RAW_DATASETS_ENDPOINT;
  const PROCESSED_DATASETS_ENDPOINT =
    process.env.REACT_APP_PROCESSED_DATASETS_ENDPOINT;
  const DELETE_RAW_ENDPOINT = process.env.REACT_APP_DELETE_RAW_ENDPOINT;
  const DELETE_PROCESSED_ENDPOINT =
    process.env.REACT_APP_DELETE_PROCESSED_ENDPOINT;
  const RAW_OVERVIEW_PATH = "/raw-dataset-overview"; // path to navigate to raw dataset overview component
  const PROCESSED_OVERVIEW_PATH = "/processed-dataset-overview"; // path to navigate to processed dataset overview component
  const PROCESSING_GUIDELINES_PATH = "/preprocessing-docs";

  const [selectedFolder, setSelectedFolder] = useState("raw");
  const [rawDatasets, setRawDatasets] = useState([]);
  const [processedDatasets, setProcessedDatasets] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const fetchData = async (folderType, page = 1) => {
    setLoading(true);
    try {
      const endpoint =
        folderType === "raw"
          ? RAW_DATASETS_ENDPOINT
          : PROCESSED_DATASETS_ENDPOINT;

      const response = await axios.get(endpoint, {
        params: {
          skip: (page - 1) * PAGE_SIZE,
          limit: PAGE_SIZE,
        },
      });

      if (folderType === "raw") {
        setRawDatasets(response.data);
      } else {
        setProcessedDatasets(response.data);
      }

      setTotalCount(response.data.length);
    } catch (err) {
      setError("Failed to load datasets. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData(selectedFolder, currentPage);
  }, [selectedFolder, currentPage]);

  const handleDelete = async (datasetId, isRaw) => {
    if (!window.confirm("Permanently delete this dataset?")) return;

    try {
      const endpoint = isRaw ? DELETE_RAW_ENDPOINT : DELETE_PROCESSED_ENDPOINT;
      const params = isRaw
        ? { fileName: datasetId }
        : { dataset_id: datasetId };

      await axios.delete(endpoint, { params });
      fetchData(selectedFolder, currentPage);
    } catch (err) {
      setError("Deletion failed. Please try again.");
    }
  };

  const handlePageChange = (newPage) => {
    if (newPage < 1) return;
    if (totalCount < PAGE_SIZE && newPage > currentPage) return;
    setCurrentPage(newPage);
  };

  const DirectorySection = ({ datasets, isRaw }) => (
    <div className="mb-8 bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {datasets.map((dataset) => (
          <FileCard
            key={isRaw ? dataset.filename : dataset.dataset_id}
            dataset={dataset}
            isRaw={isRaw}
            onDelete={handleDelete}
            onClick={() => {
              const path = isRaw
                ? `${RAW_OVERVIEW_PATH}/${encodeURIComponent(dataset.filename)}`
                : `${PROCESSED_OVERVIEW_PATH}/${encodeURIComponent(
                    dataset.filename
                  )}`;
              navigate(path);
            }}
          />
        ))}
      </div>
    </div>
  );

  const FileCard = ({ dataset, isRaw, onDelete, onClick }) => {
    const filename = isRaw ? dataset.filename : dataset.filename;
    const isProcessing = filename.endsWith("__PROCESSING__");
    const displayName = filename.replace(/__PROCESSING__$/, "");

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
            {!isRaw && (
              <p className="text-sm text-gray-500 mt-1">
                ID: {dataset.dataset_id}
              </p>
            )}
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
                onDelete(isRaw ? dataset.filename : dataset.dataset_id, isRaw);
              }}
            />
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto flex gap-8">
        {/* Sidebar Navigation */}
        <div className="w-64 space-y-2">
          <button
            className={`w-full p-3 rounded-lg text-left flex items-center gap-2 ${
              selectedFolder === "raw"
                ? "bg-blue-100 text-blue-700"
                : "hover:bg-gray-100"
            }`}
            onClick={() => {
              setSelectedFolder("raw");
              setCurrentPage(1);
            }}
          >
            <FolderIcon className="h-5 w-5" />
            Raw Uploads
          </button>
          <button
            className={`w-full p-3 rounded-lg text-left flex items-center gap-2 ${
              selectedFolder === "processed"
                ? "bg-blue-100 text-blue-700"
                : "hover:bg-gray-100"
            }`}
            onClick={() => {
              setSelectedFolder("processed");
              setCurrentPage(1);
            }}
          >
            <FolderIcon className="h-5 w-5" />
            Processed Datasets
          </button>
        </div>

        {/* Main Content */}
        <div className="flex-1 space-y-8">
          {/* Header */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <DocumentTextIcon className="h-8 w-8 text-blue-500" />
                {selectedFolder === "raw"
                  ? "Raw Uploads"
                  : "Processed Datasets"}
              </h1>

              <a
                href={PROCESSING_GUIDELINES_PATH}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
              >
                <ArrowUpTrayIcon className="h-5 w-5" />
                View Processing Guidelines
              </a>
            </div>
          </div>

          {/* Error Message */}
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

          {/* Loading State */}
          {loading && (
            <div className="text-center p-8">
              <Cog6ToothIcon className="h-8 w-8 animate-spin text-gray-400 mx-auto" />
              <p className="mt-2 text-gray-500">Loading datasets...</p>
            </div>
          )}

          {/* Dataset List */}
          {!loading && (
            <>
              <DirectorySection
                datasets={
                  selectedFolder === "raw" ? rawDatasets : processedDatasets
                }
                isRaw={selectedFolder === "raw"}
              />

              {/* Pagination Controls */}
              <div className="flex justify-center gap-4">
                <button
                  className="px-4 py-2 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 disabled:opacity-50"
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                >
                  <ChevronLeftIcon className="h-5 w-5" />
                </button>

                <span className="px-4 py-2 text-gray-700">
                  Page {currentPage}
                </span>

                <button
                  className="px-4 py-2 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 disabled:opacity-50"
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={totalCount < PAGE_SIZE}
                >
                  <ChevronRightIcon className="h-5 w-5" />
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ViewAllDatasets;

// import { useEffect, useState } from "react";
// import { useNavigate } from "react-router-dom";
// import axios from "axios";
// import {
//   TrashIcon,
//   Cog6ToothIcon,
//   DocumentTextIcon,
//   FolderIcon,
//   XCircleIcon,
//   ArrowUpTrayIcon,
// } from "@heroicons/react/24/solid";

// const REACT_APP_LIST_ALL_DATASETS_URL =
//   process.env.REACT_APP_LIST_ALL_DATASETS_URL;
// const REACT_APP_DELETE_DATAFILE_URL = process.env.REACT_APP_DELETE_DATAFILE_URL;
// const REACT_APP_DATASET_OVERVIEW_PATH = "/dataset-overview";
// const REACT_APP_PROCESSING_GUIDELINES_PATH = "/preprocessing-docs";

// const ViewAllDatasets = () => {
//   const [contents, setContents] = useState({});
//   const [error, setError] = useState(null);
//   const navigate = useNavigate();

//   useEffect(() => {
//     const fetchFiles = async () => {
//       try {
//         const response = await axios.get(REACT_APP_LIST_ALL_DATASETS_URL);
//         setContents(response.data.contents || {});
//       } catch (err) {
//         setError("Failed to load datasets. Please refresh the page.");
//         console.error(err);
//       }
//     };
//     fetchFiles();
//   }, []);

//   const handleDelete = async (file, directory) => {
//     if (!window.confirm(`Permanently delete ${file}?`)) return;

//     try {
//       await axios.delete(REACT_APP_DELETE_DATAFILE_URL, {
//         params: { directory, fileName: file },
//       });

//       setContents((prev) => ({
//         ...prev,
//         [directory]: prev[directory].filter((f) => f !== file),
//       }));
//     } catch (err) {
//       setError("Deletion failed. Ensure file isn't being processed.");
//       console.error(err);
//     }
//   };

//   const DirectorySection = ({ name, files }) => (
//     <div className="mb-8 bg-white rounded-xl p-6 shadow-sm border border-gray-100">
//       <div className="flex items-center gap-3 mb-4">
//         <FolderIcon className="h-6 w-6 text-blue-500" />
//         <h2 className="text-xl font-semibold text-gray-800">
//           {name.replace(/^\w/, (c) => c.toUpperCase())}
//         </h2>
//         <span className="text-sm text-gray-500 ml-2">
//           ({files.length} file{files.length !== 1 ? "s" : ""})
//         </span>
//       </div>

//       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
//         {files.map((file) => (
//           <FileCard
//             key={file}
//             file={file}
//             directory={name}
//             onDelete={handleDelete}
//             onClick={() =>
//               navigate(
//                 `${REACT_APP_DATASET_OVERVIEW_PATH}/${encodeURIComponent(
//                   name
//                 )}/${encodeURIComponent(file)}`
//               )
//             }
//           />
//         ))}
//       </div>
//     </div>
//   );

//   const FileCard = ({ file, directory, onDelete, onClick }) => {
//     const isProcessing = file.endsWith("__PROCESSING__");
//     const displayName = file.replace(/__PROCESSING__$/, "");

//     return (
//       <div
//         className={`group relative p-4 rounded-lg border transition-all
//           ${
//             isProcessing
//               ? "border-yellow-200 bg-yellow-50 cursor-wait"
//               : "border-gray-200 hover:border-blue-200 hover:bg-blue-50 cursor-pointer"
//           }
//         `}
//         onClick={!isProcessing ? onClick : undefined}
//       >
//         <div className="flex justify-between items-start">
//           <div className="min-w-0">
//             <p
//               className="text-gray-700 font-medium truncate"
//               title={displayName}
//             >
//               {displayName}
//             </p>
//             {isProcessing && (
//               <div className="text-sm text-yellow-600 mt-2 flex items-center">
//                 <Cog6ToothIcon className="h-4 w-4 mr-2 animate-spin" />
//                 Processing...
//               </div>
//             )}
//           </div>

//           {!isProcessing && (
//             <TrashIcon
//               className="h-5 w-5 text-red-400 hover:text-red-600 shrink-0 ml-2"
//               onClick={(e) => {
//                 e.stopPropagation();
//                 onDelete(file, directory);
//               }}
//             />
//           )}
//         </div>
//       </div>
//     );
//   };

//   return (
//     <div className="min-h-screen bg-gray-50 p-6">
//       <div className="max-w-7xl mx-auto space-y-8">
//         {/* Header Section */}
//         <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
//           <div className="flex items-center justify-between mb-4">
//             <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
//               <DocumentTextIcon className="h-8 w-8 text-blue-500" />
//               Dataset Manager
//             </h1>

//             <a
//               href={REACT_APP_PROCESSING_GUIDELINES_PATH}
//               target="_blank"
//               rel="noopener noreferrer"
//               className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
//             >
//               <ArrowUpTrayIcon className="h-5 w-5" />
//               View Processing Guidelines
//             </a>
//           </div>

//           {error && (
//             <div className="bg-red-50 p-4 rounded-lg flex items-center gap-3">
//               <XCircleIcon className="h-5 w-5 text-red-500" />
//               <span className="text-red-600">{error}</span>
//               <button
//                 onClick={() => setError(null)}
//                 className="ml-auto text-red-600 hover:text-red-800"
//               >
//                 Dismiss
//               </button>
//             </div>
//           )}
//         </div>

//         {/* Files Display */}
//         <div className="space-y-8">
//           {Object.keys(contents).length > 0 ? (
//             Object.entries(contents)
//               .sort(([a], [b]) => a.localeCompare(b))
//               .map(([directory, files]) => (
//                 <DirectorySection
//                   key={directory}
//                   name={directory}
//                   files={files.sort((a, b) => a.localeCompare(b))}
//                 />
//               ))
//           ) : (
//             <div className="text-center py-12 bg-white rounded-xl border border-gray-200">
//               <p className="text-gray-500 mb-4">No datasets found</p>
//               <button
//                 onClick={() => navigate("/ManageData")}
//                 className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors"
//               >
//                 Manage Data Page
//               </button>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default ViewAllDatasets;
