// NumericColumnDetails.js
import React from "react";
import QuartilePlot from "../GraphComponents/QuartilePlot";
import HistogramPlot from "../GraphComponents/HistogramPlot";
import {
  ChartBarIcon,
  CalculatorIcon,
  PresentationChartLineIcon,
  UserGroupIcon,
  InformationCircleIcon,
} from "@heroicons/react/24/solid";

const NumericColumn = ({ column }) => {
  const stats = [
    {
      label: "Entries",
      value: column.entries ?? "N/A",
      icon: <UserGroupIcon className="w-5 h-5 text-blue-500" />,
    },
    {
      label: "Null Values",
      value: column.nullCount ?? 0,
      icon: <InformationCircleIcon className="w-5 h-5 text-amber-500" />,
    },
    {
      label: "Mean",
      value: Number(column.mean ?? 0).toFixed(2),
      icon: <CalculatorIcon className="w-5 h-5 text-emerald-500" />,
    },
    {
      label: "Std Dev",
      value: Number(column.stddev ?? 0).toFixed(2),
      icon: <ChartBarIcon className="w-5 h-5 text-purple-500" />,
    },
  ];

  return (
    <div className="w-full p-6 grid gap-6 lg:grid-cols-2 md:grid-cols-1 bg-gray-50 rounded-xl">
      {/* Summary Section */}
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <PresentationChartLineIcon className="w-6 h-6 text-slate-600" />
            Column Statistics
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {stats.map(({ label, value, icon }) => (
              <div
                key={label}
                className="p-4 bg-white rounded-lg border border-gray-100 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gray-100 rounded-lg">{icon}</div>
                  <div>
                    <div className="text-xs font-medium text-slate-500">
                      {label}
                    </div>
                    <div className="text-lg font-mono font-semibold text-slate-800 mt-1">
                      {typeof value === "number"
                        ? value.toLocaleString()
                        : value}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quartile Graph */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <h3 className="text-lg font-semibold text-slate-800">
              Value Distribution
            </h3>
            <InformationCircleIcon className="w-5 h-5 text-blue-500" />
          </div>
          <QuartilePlot
            quartiles={{
              min: column.min,
              max: column.max,
              Q1: column.quartiles?.Q1,
              median: column.quartiles?.median,
              Q3: column.quartiles?.Q3,
            }}
          />
        </div>
      </div>

      {/* Histogram Section */}
      <div className="space-y-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <ChartBarIcon className="w-6 h-6 text-slate-600" />
          Value Distribution
        </h3>
        {column.histogram ? (
          <HistogramPlot
            bins={column.histogram.bins}
            counts={column.histogram.counts}
            xTitle={column.name}
            className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm"
          />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400 text-sm">
            <InformationCircleIcon className="w-5 h-5 mr-2" />
            No histogram data available
          </div>
        )}
      </div>
    </div>
  );
};

export default NumericColumn;
