import { useEffect, useState } from "react";
import axios from "axios";
import {
  ArrowRightCircleIcon,
  TrashIcon,
  ArrowPathIcon,
  Cog6ToothIcon,
  DocumentTextIcon,
} from "@heroicons/react/24/solid";

const ViewRecentUploads = () => {
  const [contents, setContents] = useState([]);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        `${process.env.REACT_APP_SERVER_BASE_URL}/file-upload/list-files`
      );
      const key = Object.keys(response.data.contents)[0];
      setContents(response.data.contents[key] || []);
      setError(null);
    } catch (err) {
      setError("Error fetching files. Please try again later.");
      console.error("Error fetching files: ", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const isProcessingOrCopying = (filename) => {
    return (
      filename.endsWith("_COPYING_") || filename.endsWith("__PROCESSING__")
    );
  };

  const handleDataMove = async (file) => {
    const confirmMove = window.confirm(
      `Are you sure you want to start processing '${file}'?`
    );
    if (!confirmMove) return;

    try {
      await axios.post(
        `${process.env.REACT_APP_SERVER_BASE_URL}/create-new-dataset`,
        {
          fileName: file,
        }
      );
      setSuccess(`Processing started for '${file}'`);
      setError(null);
      // Refresh the list after a short delay
      setTimeout(() => {
        fetchFiles();
        setSuccess(null);
      }, 2000);
    } catch (err) {
      setError("Error processing the file. Please try again later.");
      setSuccess(null);
      console.error("Error in processing the file: ", err);
    }
  };

  const handleDelete = async (file) => {
    const confirmDelete = window.confirm(
      `Are you sure you want to delete '${file}'? This action cannot be undone.`
    );
    if (!confirmDelete) return;

    try {
      await axios.delete(
        `${process.env.REACT_APP_SERVER_BASE_URL}/file-upload/delete/${file}`
      );
      setSuccess(`File '${file}' deleted successfully`);
      setError(null);
      // Remove from list immediately
      setContents((prevContents) =>
        prevContents.filter((f) => f.filename !== file)
      );
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError("Error deleting the file. Please try again later.");
      setSuccess(null);
      console.error("Error in deleting the file: ", err);
    }
  };

  // Format file size to human readable format
  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white p-6 rounded-xl shadow-sm border mb-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <DocumentTextIcon className="h-8 w-8 text-blue-500" />
              Recent Uploads
            </h1>
            <button
              onClick={fetchFiles}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Refreshing...
                </>
              ) : (
                <>
                  <ArrowPathIcon className="h-4 w-4" />
                  Refresh
                </>
              )}
            </button>
          </div>
        </div>

        {/* Messages */}
        {error && (
          <div className="bg-red-50 p-4 rounded-lg flex items-center gap-3 mb-6">
            <span className="text-red-600">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              ×
            </button>
          </div>
        )}

        {success && (
          <div className="bg-green-50 p-4 rounded-lg flex items-center gap-3 mb-6">
            <span className="text-green-600">{success}</span>
            <button
              onClick={() => setSuccess(null)}
              className="ml-auto text-green-500 hover:text-green-700"
            >
              ×
            </button>
          </div>
        )}

        {/* Content */}
        {loading && contents.length === 0 ? (
          <div className="animate-pulse space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded-xl" />
            ))}
          </div>
        ) : contents.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {contents.map(({ filename, size }) => (
              <div
                key={filename}
                className="bg-white p-5 shadow-lg rounded-xl hover:shadow-xl transition-all border border-gray-200"
              >
                <div className="flex items-start justify-between mb-2">
                  <p
                    className="text-gray-700 text-lg font-medium truncate flex-1"
                    title={filename}
                  >
                    {filename.length > 20
                      ? `${filename.substring(0, 20)}...`
                      : filename}
                  </p>
                </div>

                {/* Processing/Copying Status */}
                {filename.endsWith("_COPYING_") && (
                  <div className="text-sm text-blue-500 mb-2 flex items-center">
                    <span>Uploading</span>
                    <ArrowPathIcon className="h-5 w-5 ml-2 text-blue-500 animate-spin" />
                  </div>
                )}
                {filename.endsWith("__PROCESSING__") && (
                  <div className="text-sm text-yellow-500 mb-2 flex items-center">
                    <span>Processing</span>
                    <Cog6ToothIcon className="h-5 w-5 ml-2 text-yellow-500 animate-spin" />
                  </div>
                )}

                <p className="text-sm text-gray-500 mb-3">
                  Size: {typeof size === "number" ? formatFileSize(size) : size}
                </p>

                {/* Action Buttons */}
                <div className="flex justify-end gap-3 pt-3 border-t border-gray-100">
                  {!isProcessingOrCopying(filename) && (
                    <>
                      <button
                        onClick={() => handleDataMove(filename)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors text-sm"
                        title="Process this file"
                      >
                        <ArrowRightCircleIcon className="h-4 w-4" />
                        Process
                      </button>
                      <button
                        onClick={() => handleDelete(filename)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors text-sm"
                        title="Delete this file"
                      >
                        <TrashIcon className="h-4 w-4" />
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-white p-12 rounded-xl shadow-sm border text-center">
            <DocumentTextIcon className="h-16 w-16 mx-auto mb-4 text-gray-300" />
            <p className="text-gray-500 text-lg font-medium">No files found</p>
            <p className="text-gray-400 text-sm mt-2">
              Upload files using the "Add New Dataset" tab
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ViewRecentUploads;
