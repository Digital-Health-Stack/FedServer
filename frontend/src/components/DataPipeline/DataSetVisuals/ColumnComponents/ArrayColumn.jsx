// components/ColumnDetails/ArrayColumnDetails.js
import React from "react";
import {
  InformationCircleIcon,
  CubeIcon,
  ArrowsPointingOutIcon,
  ChartBarIcon,
  CalculatorIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/solid";

const ArrayColumn = ({ column }) => {
  // Extract statistics with safe defaults
  const {
    entries = 0,
    nullCount = 0,
    nonEmptyCount = 0,
    Shape = [],
    LengthStats = {},
    valueStats = {},
  } = column;

  // Calculate key metrics
  const nullPercentage = entries ? ((nullCount / entries) * 100).toFixed(1) : 0;
  const emptyPercentage = entries
    ? (((entries - nullCount - nonEmptyCount) / entries) * 100).toFixed(1)
    : 0;
  const validPercentage = entries
    ? ((nonEmptyCount / entries) * 100).toFixed(1)
    : 0;

  // Determine consistency flags
  const isUniformShape =
    Shape.length > 0 &&
    LengthStats.std === 0 &&
    LengthStats.min === LengthStats.max;

  const hasValueStats = valueStats && typeof valueStats.min === "number";
  const isNormalized =
    hasValueStats && valueStats.min >= 0 && valueStats.max <= 1;

  // Sampling information
  const sampleSize = column.sampleSize || "20% (up to 100k)";

  return (
    <div className="w-full p-6 bg-gray-50 rounded-xl space-y-6">
      {/* Metadata Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Total Entries"
          value={entries.toLocaleString()}
          icon={<CubeIcon className="w-5 h-5 text-blue-500" />}
        />
        <StatCard
          label="Null Values"
          value={`${nullCount} (${nullPercentage}%)`}
          icon={<InformationCircleIcon className="w-5 h-5 text-amber-500" />}
          warning={nullPercentage > 5}
        />
        <StatCard
          label="Valid Images"
          value={`${nonEmptyCount} (${validPercentage}%)`}
          icon={<CubeIcon className="w-5 h-5 text-green-500" />}
        />
        <StatCard
          label="Image Shape"
          value={Shape.length ? `${Shape.join(" × ")}` : "Unknown"}
          icon={<ArrowsPointingOutIcon className="w-5 h-5 text-purple-500" />}
          highlight={isUniformShape}
        />
      </div>

      {/* Key Insights */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <h3 className="font-semibold text-slate-800 mb-2 flex items-center gap-1">
          <InformationCircleIcon className="w-5 h-5 text-blue-500" />
          Preprocessing Insights
        </h3>

        <ul className="space-y-2 text-sm text-slate-700">
          <li className="flex items-start">
            <span className="inline-block w-3 h-3 rounded-full bg-green-400 mt-1.5 mr-2"></span>
            <span>
              <span className="font-medium">Consistent Dimensions:</span>{" "}
              {isUniformShape
                ? `All images are ${Shape[0]}×${Shape[1]} pixels with ${Shape[2]} channels`
                : `Images vary in size (min: ${LengthStats.min}, max: ${LengthStats.max}) - resizing needed`}
            </span>
          </li>

          {hasValueStats ? (
            <>
              <li className="flex items-start">
                <span className="inline-block w-3 h-3 rounded-full bg-blue-400 mt-1.5 mr-2"></span>
                <span>
                  <span className="font-medium">Pixel Values:</span> Range from{" "}
                  {valueStats.min.toFixed(4)} to {valueStats.max.toFixed(4)}{" "}
                  (mean: {valueStats.mean.toFixed(4)})
                  {isNormalized
                    ? " - already normalized"
                    : " - consider normalization"}
                </span>
              </li>
              <li className="flex items-start">
                <span className="inline-block w-3 h-3 rounded-full bg-amber-400 mt-1.5 mr-2"></span>
                <span>
                  <span className="font-medium">Sparsity:</span>{" "}
                  {(valueStats.sparsity * 100).toFixed(1)}% zero values -
                  {valueStats.sparsity > 0.1
                    ? " may indicate background dominance"
                    : " acceptable level"}
                </span>
              </li>
            </>
          ) : (
            <li className="flex items-start">
              <span className="inline-block w-3 h-3 rounded-full bg-amber-400 mt-1.5 mr-2"></span>
              <span>
                Pixel statistics not available - non-numeric data or sampling
                failed
              </span>
            </li>
          )}

          {nullCount > 0 && (
            <li className="flex items-start">
              <span className="inline-block w-3 h-3 rounded-full bg-red-400 mt-1.5 mr-2"></span>
              <span>
                <span className="font-medium">Null Values:</span> {nullCount} (
                {nullPercentage}%) missing images - consider imputation or
                removal
              </span>
            </li>
          )}
        </ul>
      </div>

      {/* Technical Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <h3 className="font-semibold text-slate-800 mb-2 flex items-center gap-1">
            <ChartBarIcon className="w-5 h-5 text-slate-600" />
            Length Statistics
          </h3>
          <div className="grid grid-cols-2 gap-2">
            <CompactStat label="Min" value={LengthStats.min} />
            <CompactStat label="Max" value={LengthStats.max} />
            <CompactStat label="Mean" value={LengthStats.mean?.toFixed(2)} />
            <CompactStat label="Std Dev" value={LengthStats.std?.toFixed(2)} />
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Outer array length distribution (dimension 0)
          </p>
        </div>

        {hasValueStats && (
          <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
            <h3 className="font-semibold text-slate-800 mb-2 flex items-center gap-1">
              <CalculatorIcon className="w-5 h-5 text-slate-600" />
              Pixel Value Statistics
            </h3>
            <div className="grid grid-cols-2 gap-2">
              <CompactStat label="Min" value={valueStats.min?.toFixed(4)} />
              <CompactStat label="Max" value={valueStats.max?.toFixed(4)} />
              <CompactStat label="Mean" value={valueStats.mean?.toFixed(4)} />
              <CompactStat
                label="Median"
                value={valueStats.median?.toFixed(4)}
              />
              <CompactStat label="Std Dev" value={valueStats.std?.toFixed(4)} />
              <CompactStat
                label="Sparsity"
                value={`${(valueStats.sparsity * 100).toFixed(1)}%`}
              />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Calculated from {sampleSize} sample pixels
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// Stat card component
const StatCard = ({ label, value, icon, warning, highlight }) => (
  <div
    className={`p-3 rounded-lg border ${
      warning
        ? "bg-amber-50 border-amber-200"
        : highlight
          ? "bg-green-50 border-green-200"
          : "bg-white border-gray-200"
    }`}
  >
    <div className="flex items-center gap-2">
      {icon}
      <div>
        <div className="text-xs font-medium text-slate-500">{label}</div>
        <div
          className={`text-base font-semibold ${
            warning
              ? "text-amber-700"
              : highlight
                ? "text-green-700"
                : "text-slate-800"
          }`}
        >
          {value}
        </div>
      </div>
    </div>
  </div>
);

// Compact stat display
const CompactStat = ({ label, value }) => (
  <div className="flex justify-between items-center py-1 border-b border-gray-100">
    <span className="text-xs text-slate-500">{label}:</span>
    <span className="text-sm font-mono font-medium text-slate-800">
      {value || "N/A"}
    </span>
  </div>
);

export default ArrayColumn;
