import React from "react";
import NumericColumn from "./ColumnComponents/NumericColumn.jsx";
import StringColumn from "./ColumnComponents/StringColumn.jsx";

const ColumnDetails = ({ columnStats }) => {
  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Column Details</h1>
      <div className="space-y-6">
        {columnStats.map((col, index) => (
          <div
            key={col.name}
            className="border rounded-lg shadow-md bg-white p-4"
          >
            {/* Column Banner */}
            <div className="flex justify-between items-center mb-4">
              <div>
                <h3 className="font-bold text-xl text-gray-800">
                  Column {index + 1}: {col.name}
                </h3>
                <p className="text-sm text-gray-600">Type: {col.type}</p>
              </div>
            </div>

            {/* Render Numeric or String Column Details */}
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
    </div>
  );
};

export default ColumnDetails;
