import React, { useState } from "react";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  BookmarkIcon,
  BookmarkSlashIcon,
  ChartBarIcon,
  ArrowsPointingInIcon,
} from "@heroicons/react/24/outline";
import NumericColumn from "../ColumnComponents/NumericColumn";
import StringColumn from "../ColumnComponents/StringColumn";
import ArrayColumn from "../ColumnComponents/ArrayColumn";

const columnComponents = {
  Numeric: NumericColumn,
  String: StringColumn,
  Array: ArrayColumn,
};

const ColumnDetails = ({ columnStats }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [pinnedColumns, setPinnedColumns] = useState([]);
  const [isExpanded, setIsExpanded] = useState(true);

  const handleNavigation = (direction) => {
    setCurrentIndex(
      (prev) => (prev + direction + columnStats.length) % columnStats.length
    );
  };

  const handlePin = () => {
    const currentColumn = columnStats[currentIndex];
    if (pinnedColumns.some((col) => col.name === currentColumn.name)) {
      setPinnedColumns(
        pinnedColumns.filter((col) => col.name !== currentColumn.name)
      );
    } else {
      setPinnedColumns([...pinnedColumns, currentColumn]);
    }
  };

  if (columnStats.length === 0) return null;
  const currentColumn = columnStats[currentIndex];
  let col_type = "Unknown";

  if (currentColumn.type.includes("ArrayType")) {
    col_type = "Array";
  } else if (currentColumn.type.includes("StringType")) {
    col_type = "String";
  } else if (
    currentColumn.type.includes("IntegerType") ||
    currentColumn.type.includes("DoubleType") ||
    currentColumn.type.includes("FloatType") ||
    currentColumn.type.includes("LongType") ||
    currentColumn.type.includes("ShortType")
  ) {
    col_type = "Numeric";
  }

  const ColumnComponent = columnComponents[col_type];
  const isPinned = pinnedColumns.some((col) => col.name === currentColumn.name);

  return (
    <div className="bg-white rounded-xl shadow-sm p-2">
      <div className="p-4 border-b flex items-center justify-between bg-blue-50 rounded-t-xl">
        <div className="flex items-center gap-2">
          <ChartBarIcon className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-blue-800">
            Column Analysis
          </h2>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 hover:bg-blue-100 rounded-full"
        >
          <ArrowsPointingInIcon className="w-5 h-5 text-blue-600" />
        </button>
      </div>

      {isExpanded && (
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex items-center gap-4 mb-4">
            <button
              onClick={() => handleNavigation(-1)}
              className="p-2 hover:bg-gray-100 rounded-full"
            >
              <ChevronLeftIcon className="w-6 h-6 text-gray-600" />
            </button>

            <select
              value={currentColumn.name}
              onChange={(e) => {
                const index = columnStats.findIndex(
                  (col) => col.name === e.target.value
                );
                setCurrentIndex(index);
              }}
              className="flex-1 p-2 bg-white border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {columnStats.map((col, index) => (
                <option key={col.name} value={col.name}>
                  {`${index + 1}. ${col.name} (${col.type.replace(
                    "Type()",
                    ""
                  )})`}
                </option>
              ))}
            </select>

            <button
              onClick={() => handleNavigation(1)}
              className="p-2 hover:bg-gray-100 rounded-full"
            >
              <ChevronRightIcon className="w-6 h-6 text-gray-600" />
            </button>
          </div>

          <div className="bg-gray-50 rounded-xl p-4 mb-4">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="font-medium text-gray-900 text-lg">
                  {currentColumn.name}
                </h3>
                <p className="text-sm text-gray-500">{currentColumn.type}</p>
              </div>
              <button
                onClick={handlePin}
                className={`p-1 rounded-full ${
                  isPinned ? "text-blue-600" : "text-gray-400"
                }`}
              >
                {isPinned ? (
                  <BookmarkSlashIcon className="w-6 h-6 fill-current" />
                ) : (
                  <BookmarkIcon className="w-6 h-6" />
                )}
              </button>
            </div>

            <div className="border-t pt-4">
              {ColumnComponent ? (
                <ColumnComponent column={currentColumn} />
              ) : (
                <div className="text-gray-500 text-sm p-2">
                  No visualization component available for {currentColumn.type}
                </div>
              )}
            </div>
          </div>

          {pinnedColumns.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-700">
                Pinned Columns
              </h3>
              {pinnedColumns.map((col) => {
                const PinnedComponent = columnComponents[col.type];
                return (
                  <div key={col.name} className="bg-gray-50 rounded-xl p-4">
                    <div className="flex justify-between items-center mb-2">
                      <div>
                        <h4 className="font-medium text-gray-900">
                          {col.name}
                        </h4>
                        <p className="text-sm text-gray-500">{col.type}</p>
                      </div>
                      <button
                        onClick={() =>
                          setPinnedColumns(
                            pinnedColumns.filter((c) => c.name !== col.name)
                          )
                        }
                        className="text-red-400 hover:text-red-600"
                      >
                        <BookmarkSlashIcon className="w-5 h-5" />
                      </button>
                    </div>
                    <div className="border-t pt-4">
                      {PinnedComponent ? (
                        <PinnedComponent column={col} />
                      ) : (
                        <div className="text-gray-500 text-sm p-2">
                          No visualization component available for {col.type}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ColumnDetails;
