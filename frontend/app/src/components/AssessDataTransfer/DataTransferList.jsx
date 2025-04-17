// DataTransferList.jsx
import { useState } from "react";
import {
  CircleStackIcon,
  ClockIcon,
  ArrowPathIcon,
} from "@heroicons/react/24/outline";
import Spinner from "./Spinner";

const DataTransferList = ({
  transfers,
  selectedId,
  onSelect,
  loadMore,
  hasMore,
}) => {
  const [loadingMore, setLoadingMore] = useState(false);

  const handleLoadMore = async () => {
    setLoadingMore(true);
    await loadMore();
    setLoadingMore(false);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Scrollable List Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="divide-y divide-gray-100">
          {transfers.map((transfer) => (
            <button
              key={transfer.id}
              onClick={() => onSelect(transfer.id)}
              className={`w-full p-4 text-left transition-colors duration-200 ${
                selectedId === transfer.id
                  ? "bg-blue-600/5 border-l-4 border-blue-600"
                  : "hover:bg-gray-50"
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900">
                    {transfer.training_name}
                  </h3>
                  <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
                    <CircleStackIcon className="w-4 h-4" />
                    <span>
                      {transfer.num_datapoints.toLocaleString()} points
                    </span>
                  </div>
                </div>
                <div className="text-xs text-gray-400 flex items-center gap-1">
                  <ClockIcon className="w-3.5 h-3.5" />
                  {new Date(transfer.transferredAt).toLocaleDateString()}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Load More Button */}
      {hasMore && (
        <div className="sticky bottom-0 border-t border-gray-100 bg-white/95 backdrop-blur-sm">
          <button
            onClick={handleLoadMore}
            disabled={loadingMore}
            className="w-full p-3 text-sm font-medium text-blue-600 hover:bg-gray-50 flex items-center justify-center gap-2"
          >
            {loadingMore ? (
              <Spinner className="w-4 h-4 text-blue-600" />
            ) : (
              <>
                <ArrowPathIcon className="w-4 h-4" />
                Load More
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default DataTransferList;
