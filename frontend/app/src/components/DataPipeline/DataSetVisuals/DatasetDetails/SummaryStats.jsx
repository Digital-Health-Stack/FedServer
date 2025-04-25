import React from "react";
import { InformationCircleIcon } from "@heroicons/react/24/outline";

const SummaryStats = ({ filename, numRows, numCols }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm p-2 mt-4">
      <div className="p-4 border-b flex items-center justify-between bg-blue-50 rounded-t-xl">
        <div className="flex items-center gap-2">
          <InformationCircleIcon className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-blue-800">
            Dataset Overview
          </h2>
          {/* <h1 className="text-2xl font-semibold">Dataset Overview</h1> */}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Filename</p>
          <p className="font-mono text-lg font-medium truncate">{filename}</p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Rows</p>
          <p className="text-3xl font-bold text-blue-600">{numRows}</p>
        </div>
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600 mb-1">Columns</p>
          <p className="text-3xl font-bold text-blue-600">{numCols}</p>
        </div>
      </div>
    </div>
  );
};

export default SummaryStats;
