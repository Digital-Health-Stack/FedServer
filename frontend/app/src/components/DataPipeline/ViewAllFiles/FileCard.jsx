import { useState } from "react";
import axios from "axios";
import {
  PencilIcon,
  TrashIcon,
  Cog6ToothIcon,
} from "@heroicons/react/24/solid";
import EditDatasetModal from "./EditDatasetModal";

const RAW_DATASET_RENAME_URL =
  process.env.REACT_APP_PRIVATE_SERVER_URL + "/edit-raw-dataset-details";
const PROCESSED_DATASET_RENAME_URL =
  process.env.REACT_APP_PRIVATE_SERVER_URL + "/edit-dataset-details";

const FileCard = ({ dataset, isRaw, onDelete, onClick, onEditSuccess }) => {
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isProcessing] = useState(dataset.filename.endsWith("__PROCESSING__"));
  const displayName = dataset.filename.replace(/__PROCESSING__$/, "");

  const handleEdit = async (newFilename, newDescription) => {
    try {
      const endpoint = isRaw
        ? RAW_DATASET_RENAME_URL
        : PROCESSED_DATASET_RENAME_URL;

      await axios.put(endpoint, {
        dataset_id: dataset.dataset_id,
        filename: newFilename,
        description: newDescription,
      });

      onEditSuccess();
    } catch (error) {
      console.error("Error updating dataset:", error);
    }
  };

  return (
    <div
      className={`group relative p-5 rounded-xl border transition-all
        ${
          isProcessing
            ? "border-yellow-200 bg-yellow-50 cursor-wait"
            : "border-gray-200 hover:border-blue-500 hover:bg-blue-50 hover:shadow-md cursor-pointer"
        }`}
      onClick={(e) => {
        if (!isProcessing && !isEditModalOpen) {
          onClick();
        }
      }}
    >
      <div className="flex justify-between items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-gray-800 truncate">
              {displayName.length > 30
                ? `${displayName.slice(0, 30)}...`
                : displayName}
            </h3>
            {!isRaw && (
              <span className="text-xs font-medium text-gray-500 px-2 py-1 bg-gray-100 rounded-full">
                ID: {dataset.dataset_id}
              </span>
            )}
          </div>

          {dataset.description && (
            <p className="text-sm text-gray-600 mt-2 line-clamp-3">
              {dataset.description}
            </p>
          )}

          {isProcessing && (
            <div className="mt-3 flex items-center text-sm text-yellow-700">
              <Cog6ToothIcon className="h-4 w-4 mr-2 animate-spin" />
              Processing...
            </div>
          )}
        </div>

        {!isProcessing && (
          <div className="flex flex-col gap-2">
            <button
              className="p-1.5 text-blue-400 hover:text-blue-600 hover:bg-blue-100 rounded-full transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                setIsEditModalOpen(true);
              }}
            >
              <PencilIcon className="h-5 w-5" />
            </button>
            <button
              className="p-1.5 text-red-400 hover:text-red-600 hover:bg-red-100 rounded-full transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(dataset.dataset_id, isRaw);
              }}
            >
              <TrashIcon className="h-5 w-5" />
            </button>
          </div>
        )}
      </div>

      <EditDatasetModal
        isOpen={isEditModalOpen}
        onClose={() => setIsEditModalOpen(false)}
        initialFilename={displayName}
        initialDescription={dataset.description}
        onSave={handleEdit}
      />
    </div>
  );
};

export default FileCard;
