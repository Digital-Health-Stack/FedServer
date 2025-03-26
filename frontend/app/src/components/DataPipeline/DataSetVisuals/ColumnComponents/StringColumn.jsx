import React from "react";
import BarPlot from "../GraphComponents/BarPlot";
import {
  CheckCircleIcon,
  XCircleIcon,
  SparklesIcon,
  ChartBarIcon,
} from "@heroicons/react/24/solid";

const StringColumn = ({ column }) => {
  const bins = column.topCategories.map((category) => category.value);
  const counts = column.topCategories.map((category) => category.count);

  return (
    <div className="w-full p-6 grid lg:grid-cols-8 md:grid-cols-1 gap-8">
      {/* Summary Section */}
      <div className="lg:col-span-4 space-y-6">
        <div className="flex items-center gap-2 border-b pb-4">
          <ChartBarIcon className="w-6 h-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            Column Summary
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-1 gap-4">
          <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
            <CheckCircleIcon className="w-5 h-5 text-green-600" />
            <div>
              <p className="text-sm text-gray-500">Total Entries</p>
              <p className="text-2xl font-medium">{column.entries || "N/A"}</p>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
            <XCircleIcon className="w-5 h-5 text-red-600" />
            <div>
              <p className="text-sm text-gray-500">Null Values</p>
              <p className="text-2xl font-medium">{column.nullCount}</p>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
            <SparklesIcon className="w-5 h-5 text-purple-600" />
            <div>
              <p className="text-sm text-gray-500">Unique Values</p>
              <p className="text-2xl font-medium">{column.uniqueCount}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="lg:col-span-4">
        <div className="border rounded-lg shadow-sm p-6 h-full">
          <div className="flex items-center gap-2 mb-6">
            <ChartBarIcon className="w-6 h-6 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900">
              Value Distribution
            </h3>
          </div>
          <div className="h-[300px]">
            <BarPlot bins={bins} counts={counts} xTitle={column.name} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default StringColumn;
