import { InformationCircleIcon, ChartBarIcon, TableCellsIcon, DocumentTextIcon } from "@heroicons/react/24/outline";
import StatsTable from "./StatsTable";

const DatasetInfo = ({ data }) => {
  const datasetInfo = data?.dataset_info;
  if (!datasetInfo) return null;

  return (
    <div className="bg-white shadow rounded-lg border border-gray-200 p-6 mb-6">
      <h3 className="text-xl font-semibold mb-6 text-gray-800 border-b pb-2 flex items-center">
        <InformationCircleIcon className="h-5 w-5 text-indigo-700 mr-2" />
        Dataset Information
      </h3>

      {/* Combined Dataset Details and Output Columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Dataset Details */}
        <div className="bg-gray-50 p-5 rounded-lg border border-gray-200">
          <div className="flex items-center mb-4">
            <DocumentTextIcon className="h-5 w-5 text-gray-600 mr-2" />
            <h4 className="text-lg font-medium text-gray-700">Dataset Details</h4>
          </div>
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-4 items-center">
              <span className="text-sm font-medium text-gray-500">Client Filename</span>
              <span className="col-span-2 text-sm text-gray-800 bg-gray-100 px-3 py-2 rounded">
                {datasetInfo.client_filename}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-4 items-center">
              <span className="text-sm font-medium text-gray-500">Server Filename</span>
              <span className="col-span-2 text-sm text-gray-800 bg-gray-100 px-3 py-2 rounded">
                {datasetInfo.server_filename}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-4 items-center">
              <span className="text-sm font-medium text-gray-500">Task Name</span>
              <span className="col-span-2 text-sm text-gray-800 bg-gray-100 px-3 py-2 rounded">
                {datasetInfo.task_name}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-4 items-center">
              <span className="text-sm font-medium text-gray-500">Metric</span>
              <span className="col-span-2 text-sm text-gray-800 bg-gray-100 px-3 py-2 rounded">
                {datasetInfo.metric}
              </span>
            </div>
          </div>
        </div>

        {/* Output Columns */}
        <div className="bg-gray-50 p-5 rounded-lg border border-gray-200">
          <div className="flex items-center mb-4">
            <TableCellsIcon className="h-5 w-5 text-gray-600 mr-2" />
            <h4 className="text-lg font-medium text-gray-700">Output Columns</h4>
          </div>
          <div className="flex flex-wrap gap-3">
            {datasetInfo.output_columns?.map((col, idx) => (
              <span 
                key={idx} 
                className="bg-indigo-100 text-indigo-800 text-sm font-medium px-4 py-2 rounded-md flex items-center justify-center min-w-[100px]"
              >
                {col}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div>
        <div className="flex items-center mb-4">
          <ChartBarIcon className="h-5 w-5 text-gray-600 mr-2" />
          <h4 className="text-lg font-medium text-gray-700">Dataset Statistics</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 p-5 rounded-lg border border-gray-200">
            <h5 className="font-medium text-gray-600 mb-3">Client Statistics</h5>
            <StatsTable stats={datasetInfo.client_stats} />
          </div>
          <div className="bg-gray-50 p-5 rounded-lg border border-gray-200">
            <h5 className="font-medium text-gray-600 mb-3">Server Statistics</h5>
            <StatsTable stats={datasetInfo.server_stats} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetInfo;