import React, { useState } from "react";
import NumericColumn from "../ColumnComponents/NumericColumn.jsx";
import StringColumn from "../ColumnComponents/StringColumn.jsx";
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  BookmarkIcon,
} from "@heroicons/react/24/solid";

const ColumnDetails = ({ columnStats }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [pinnedColumns, setPinnedColumns] = useState([]);

  const unpinnedColumns = columnStats.filter(
    (col) => !pinnedColumns.some((pinned) => pinned.name === col.name)
  );

  const handleNext = () => {
    setCurrentIndex((prev) => (prev + 1) % unpinnedColumns.length);
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) =>
      prev === 0 ? unpinnedColumns.length - 1 : prev - 1
    );
  };

  const togglePin = (column) => {
    if (pinnedColumns.some((c) => c.name === column.name)) {
      setPinnedColumns(pinnedColumns.filter((c) => c.name !== column.name));
    } else {
      setPinnedColumns([...pinnedColumns, column]);
    }
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Column Details</h1>

      {/* Carousel Section */}
      {unpinnedColumns.length > 0 && (
        <div className="relative mb-8">
          <div className="absolute inset-y-0 left-0 flex items-center -ml-12">
            <button
              onClick={handlePrevious}
              className="p-2 bg-white rounded-full shadow-md hover:bg-gray-50"
            >
              <ArrowLeftIcon className="h-6 w-6 text-gray-600" />
            </button>
          </div>

          <div className="border rounded-lg shadow-md bg-white p-4 mx-4">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h3 className="font-bold text-xl text-gray-800">
                  Column {currentIndex + 1}:{" "}
                  {unpinnedColumns[currentIndex]?.name}
                </h3>
                <p className="text-sm text-gray-600">
                  Type: {unpinnedColumns[currentIndex]?.type}
                </p>
              </div>
              <button
                onClick={() => togglePin(unpinnedColumns[currentIndex])}
                className="p-2 hover:bg-gray-100 rounded-full"
              >
                <BookmarkIcon className="h-6 w-6 text-blue-600" />
              </button>
            </div>

            {[
              "IntegerType()",
              "DoubleType()",
              "FloatType()",
              "LongType()",
            ].includes(unpinnedColumns[currentIndex]?.type) && (
              <NumericColumn column={unpinnedColumns[currentIndex]} />
            )}
            {unpinnedColumns[currentIndex]?.type === "StringType()" && (
              <StringColumn column={unpinnedColumns[currentIndex]} />
            )}
          </div>

          <div className="absolute inset-y-0 right-0 flex items-center -mr-12">
            <button
              onClick={handleNext}
              className="p-2 bg-white rounded-full shadow-md hover:bg-gray-50"
            >
              <ArrowRightIcon className="h-6 w-6 text-gray-600" />
            </button>
          </div>
        </div>
      )}

      {/* Pinned Columns Section */}
      {pinnedColumns.length > 0 && (
        <div className="space-y-6 mt-8">
          <h2 className="text-2xl font-semibold text-gray-700">
            Pinned Columns
          </h2>
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
                <button
                  onClick={() => togglePin(col)}
                  className="p-2 hover:bg-gray-100 rounded-full"
                >
                  <BookmarkIcon className="h-6 w-6 text-red-600 rotate-45" />
                </button>
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

// import React from "react";
// import NumericColumn from "./ColumnComponents/NumericColumn.jsx";
// import StringColumn from "./ColumnComponents/StringColumn.jsx";

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
//             {[
//               "IntegerType()",
//               "DoubleType()",
//               "FloatType()",
//               "LongType()",
//             ].includes(col.type) && <NumericColumn column={col} />}
//             {col.type === "StringType()" && <StringColumn column={col} />}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ColumnDetails;
