// import React from "react";
// import NumericColumn from "../ColumnComponents/NumericColumn.jsx";
// import StringColumn from "../ColumnComponents/StringColumn.jsx";

// const ColumnDetails = ({ columnStats }) => {
//   return (
//     <div className="p-6 bg-gray-100 min-h-screen">
//       <h1 className="text-3xl font-bold mb-6">Column Details</h1>
//       <div className="space-y-6">
//         {columnStats.map((col, index) => (
//           <div
//             key={col.name}
//             className="border rounded-lg shadow-md bg-white p-4"
//           >
//             {/* Column Banner */}
//             <div className="flex justify-between items-center mb-4">
//               <div>
//                 <h3 className="font-bold text-xl text-gray-800">
//                   Column {index + 1}: {col.name}
//                 </h3>
//                 <p className="text-sm text-gray-600">Type: {col.type}</p>
//               </div>
//             </div>

//             {/* Render Numeric or String Column Details */}
//             {["IntegerType", "DoubleType", "FloatType", "LongType"].includes(
//               col.type
//             ) && <NumericColumn column={col} />}
//             {col.type === "StringType" && <StringColumn column={col} />}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ColumnDetails;

import React, { useState } from "react";
import NumericColumn from "../ColumnComponents/NumericColumn.jsx";
import StringColumn from "../ColumnComponents/StringColumn.jsx";

const ColumnDetails = ({ columnStats }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [pinnedColumns, setPinnedColumns] = useState([]);

  const handleNext = () => {
    setCurrentIndex((prev) => (prev + 1) % columnStats.length);
  };

  const handlePrev = () => {
    setCurrentIndex(
      (prev) => (prev - 1 + columnStats.length) % columnStats.length
    );
  };

  const handlePin = () => {
    const currentColumn = columnStats[currentIndex];
    if (!pinnedColumns.some((col) => col.name === currentColumn.name)) {
      setPinnedColumns([...pinnedColumns, currentColumn]);
    }
  };

  if (columnStats.length === 0) return null;

  const currentColumn = columnStats[currentIndex];

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Column Details</h1>

      {/* Carousel Section */}
      <div className="relative mb-8">
        <div className="flex items-center justify-between">
          <button
            onClick={handlePrev}
            className="p-2 hover:bg-gray-200 rounded-full"
          >
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>

          <div className="flex-1 mx-4">
            <div className="border rounded-lg shadow-md bg-white p-4">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="font-bold text-xl text-gray-800">
                    Column {currentIndex + 1}: {currentColumn.name}
                  </h3>
                  <p className="text-sm text-gray-600">
                    Type: {currentColumn.type}
                  </p>
                </div>
                <button
                  onClick={handlePin}
                  className="p-2 hover:bg-gray-100 rounded-full"
                  title="Pin column"
                >
                  <svg
                    className="w-6 h-6"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
                    />
                  </svg>
                </button>
              </div>

              {[
                "IntegerType()",
                "DoubleType()",
                "FloatType()",
                "LongType()",
              ].includes(currentColumn.type) && (
                <NumericColumn column={currentColumn} />
              )}
              {currentColumn.type === "StringType()" && (
                <StringColumn column={currentColumn} />
              )}
            </div>
          </div>

          <button
            onClick={handleNext}
            className="p-2 hover:bg-gray-200 rounded-full"
          >
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Pinned Columns Section */}
      {pinnedColumns.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold mb-4">Pinned Columns</h2>
          {pinnedColumns.map((col, index) => (
            <div
              key={col.name}
              className="border rounded-lg shadow-md bg-white p-4"
            >
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="font-bold text-xl text-gray-800">
                    Pinned Column {index + 1}: {col.name}
                  </h3>
                  <p className="text-sm text-gray-600">Type: {col.type}</p>
                </div>
              </div>

              {[
                "IntegerType()",
                "DoubleType()",
                "FloatType()",
                "LongType()",
              ].includes(col.type) && <NumericColumn column={col} />}
              {col.type === "StringType()" && <StringColumn column={col} />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ColumnDetails;
