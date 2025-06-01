import { useEffect, useState } from "react";
import axios from "axios";
import {
  ArrowRightCircleIcon,
  TrashIcon,
  ArrowPathIcon,
  Cog6ToothIcon,
} from "@heroicons/react/24/solid";
import {
  createNewDataset,
  viewRecentUploads,
  deleteRecentUpload,
} from "../../services/privateService";

const ViewRecentUploads = () => {
  const [contents, setContents] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await viewRecentUploads();
        const key = Object.keys(response.data.contents)[0];
        setContents(response.data.contents[key] || []);
        console.log("Files fetched successfully");
      } catch (err) {
        setError("Error fetching files. Please try again later.");
        console.error("Error fetching files: ", err);
      }
    };
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
      await createNewDataset({
        fileName: file,
      });
      setContents((prevContents) =>
        prevContents.filter((f) => f.filename !== file)
      );
      //   refresh the page after 1 sec
      setTimeout(() => window.location.reload(), 1000);
      console.log(`Dataset creation started for ${file}`);
    } catch (err) {
      setError("Error processing the file. Please try again later.");
      console.error("Error in processing the file: ", err);
    }
  };

  const handleDelete = async (dir, file) => {
    const confirmDelete = window.confirm(
      `Are you sure you want to delete '${file}'? This action cannot be undone.`
    );
    if (!confirmDelete) return;

    try {
      deleteRecentUpload({ directory: dir, filename: file });
      setContents((prevContents) =>
        prevContents.filter((f) => f.filename !== file)
      );
      //   refresh the page
      window.location.reload();
    } catch (err) {
      setError("Error deleting the file. Please try again later.");
      console.error("Error in deleting the file: ", err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">
          Recent Uploads
        </h1>
        {error && <div className="text-red-500 text-center mb-4">{error}</div>}
        {contents.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {contents.map(({ filename, size }) => (
              <div
                key={filename}
                className="bg-white p-5 shadow-lg rounded-xl cursor-pointer hover:shadow-xl hover:bg-gray-50 transition-all border border-gray-200"
              >
                <p className="text-gray-700 text-lg font-medium truncate flex justify-between items-center">
                  <span title={filename}>
                    {filename.length > 22
                      ? `${filename.substring(0, 22)}...`
                      : filename}
                  </span>
                </p>
                {/* If file name ends with __COPYING__, show 'Uploading' with cloud upload icon */}
                {filename.endsWith("_COPYING_") && (
                  <div className="text-sm text-blue-500 mt-1 flex items-center">
                    <span>Uploading</span>
                    <ArrowPathIcon className="h-5 w-5 ml-2 text-blue-500 animate-spin" />
                  </div>
                )}
                {/* If file name ends with __PROCESSING__, show 'Processing' with refresh icon */}
                {filename.endsWith("__PROCESSING__") && (
                  <div className="text-sm text-yellow-500 mt-1 flex items-center">
                    <span>Processing</span>
                    <Cog6ToothIcon className="h-5 w-5 ml-2 text-yellow-500 animate-spin" />
                  </div>
                )}

                <p className="text-sm text-gray-500 mt-1">{size}</p>
                <div className="flex justify-end mt-3 gap-3">
                  {isProcessingOrCopying(filename) || (
                    <ArrowRightCircleIcon
                      className="h-6 w-6 text-green-500 cursor-pointer hover:text-green-700"
                      onClick={() => handleDataMove(filename)}
                    />
                  )}

                  {isProcessingOrCopying(filename) || (
                    <TrashIcon
                      className="h-6 w-6 text-red-500 cursor-pointer hover:text-red-700"
                      onClick={() => handleDelete("tmpuploads", filename)}
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-500">No files found.</div>
        )}
      </div>
    </div>
  );
};

export default ViewRecentUploads;
