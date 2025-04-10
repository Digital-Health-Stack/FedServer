import React, { useEffect, useState } from "react";

const SummaryStats = ({ filename, numRows, numCols }) => {
  return (
    <div className="p-6 bg-gray-100">
      <h1 className="text-3xl font-bold mb-6 text-center">Dataset Overview</h1>
      <div className="space-y-6">
        {/* Banner-like card for dataset details */}
        <div className="bg-white shadow-lg rounded-lg p-6 flex items-center justify-between border-l-4 border-blue-500">
          <div className="space-y-4 w-full">
            <p className="text-lg text-gray-1000 flex justify-between items-center">
              <span className="font-bold text-gray-800 text-xl mr-4">
                File Name:
              </span>
              <span className="text-2xl font-bold text-green-800">
                {filename}
              </span>
            </p>
            <p className="text-lg text-gray-1000 flex justify-between items-center">
              <span className="font-bold text-gray-800 text-xl mr-4">
                Number of Rows:
              </span>
              <span className="text-5xl font-extrabold text-green-800">
                {numRows}
              </span>
            </p>
            <p className="text-lg text-gray-600 flex justify-between items-center">
              <span className="font-bold text-gray-800 text-xl mr-4">
                Number of Columns:
              </span>
              <span className="text-5xl font-extrabold text-green-800">
                {numCols}
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SummaryStats;
