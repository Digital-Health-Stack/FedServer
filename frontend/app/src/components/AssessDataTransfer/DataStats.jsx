const DataStats = ({ stats }) => {
  console.log("DataStats in QPD: ", stats);
  if (!stats || !Object.keys(stats).length) {
    return (
      <div className="text-gray-500 p-4 bg-gray-50 rounded-md">
        No statistics available
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      {Object.entries(stats).map(([key, value]) => (
        <div key={key} className="p-3 bg-gray-50 rounded-md">
          <div className="text-sm text-gray-500 capitalize">{key}</div>
          <div className="font-medium">
            {typeof value === "object" ? JSON.stringify(value) : value}
          </div>
        </div>
      ))}
    </div>
  );
};

export default DataStats;
