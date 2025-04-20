import React from "react";

const StatsTable = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <tbody className="divide-y divide-gray-200">
          <tr>
            <td className="px-2 py-1 text-gray-600 font-medium">Rows</td>
            <td className="px-2 py-1 text-gray-800">{stats.numRows}</td>
          </tr>
          <tr>
            <td className="px-2 py-1 text-gray-600 font-medium">Columns</td>
            <td className="px-2 py-1 text-gray-800">{stats.numColumns}</td>
          </tr>
          
          {stats.columnStats?.map((col, idx) => (
            <React.Fragment key={idx}>
              <tr className="bg-gray-100">
                <td colSpan="2" className="px-2 py-1 font-medium text-gray-700">
                  Column: {col.name} ({col.type})
                </td>
              </tr>
              <tr>
                <td className="px-2 py-1 text-gray-600">Entries</td>
                <td className="px-2 py-1 text-gray-800">{col.entries}</td>
              </tr>
              <tr>
                <td className="px-2 py-1 text-gray-600">Null Count</td>
                <td className="px-2 py-1 text-gray-800">{col.nullCount}</td>
              </tr>
              {col.mean !== undefined && (
                <tr>
                  <td className="px-2 py-1 text-gray-600">Mean</td>
                  <td className="px-2 py-1 text-gray-800">{col.mean.toFixed(4)}</td>
                </tr>
              )}
              {col.min !== undefined && (
                <>
                  <tr>
                    <td className="px-2 py-1 text-gray-600">Min</td>
                    <td className="px-2 py-1 text-gray-800">{col.min}</td>
                  </tr>
                  <tr>
                    <td className="px-2 py-1 text-gray-600">Max</td>
                    <td className="px-2 py-1 text-gray-800">{col.max}</td>
                  </tr>
                </>
              )}
              {col.quartiles && (
                <>
                  <tr>
                    <td className="px-2 py-1 text-gray-600">Median</td>
                    <td className="px-2 py-1 text-gray-800">{col.quartiles.median}</td>
                  </tr>
                  <tr>
                    <td className="px-2 py-1 text-gray-600">IQR</td>
                    <td className="px-2 py-1 text-gray-800">{col.quartiles.IQR}</td>
                  </tr>
                </>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default StatsTable;