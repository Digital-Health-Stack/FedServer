import React from "react";
import {
  FolderIcon,
  ClockIcon,
  SparklesIcon,
} from "@heroicons/react/24/outline";

// path is navigation path, not URL
const VIEW_RECENT_UPLOADS_PATH = "/view-recent-uploads";
const VIEW_ALL_DATASETS_PATH = "/view-all-datasets";

const ManageData = () => {
  return (
    <div className="bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12 space-y-4">
          <h1 className="text-4xl font-bold text-gray-900">
            Dataset Management Portal
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Access uploads, preprocess and manage stored datasets
          </p>
        </div>

        {/* Action Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Recent Uploads Card */}
          <div className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200 border border-gray-100">
            <div className="flex flex-col items-center text-center space-y-4">
              <FolderIcon className="h-16 w-16 text-blue-600" />
              <h3 className="text-2xl font-semibold text-gray-800">
                Raw Datasets Archive
              </h3>
              <p className="text-gray-600 mb-6">
                Access recently uploaded datasets in temporary storage
              </p>
              <a
                href={VIEW_RECENT_UPLOADS_PATH}
                className="w-full bg-blue-100 hover:bg-blue-200 text-blue-700 px-6 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                <ClockIcon className="h-5 w-5" />
                View Recent Uploads
              </a>
            </div>
          </div>

          {/* Processed Datasets Card */}
          <div className="bg-white p-8 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200 border border-gray-100">
            <div className="flex flex-col items-center text-center space-y-4">
              <SparklesIcon className="h-16 w-16 text-green-600" />
              <h3 className="text-2xl font-semibold text-gray-800">
                Manage Data
              </h3>
              <p className="text-gray-600 mb-6">
                Access uploads and preprocessed datasets ready for analysis
              </p>
              <a
                href={VIEW_ALL_DATASETS_PATH}
                className="w-full bg-green-100 hover:bg-green-200 text-green-700 px-6 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
              >
                <SparklesIcon className="h-5 w-5" />
                View Datasets
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManageData;
