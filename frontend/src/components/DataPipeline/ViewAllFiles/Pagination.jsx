import { ChevronLeftIcon, ChevronRightIcon } from "@heroicons/react/24/solid";

const Pagination = ({ currentPage, totalCount, pageSize, onPageChange }) => {
  const totalPages = Math.ceil(totalCount / pageSize);

  return (
    <div className="flex items-center justify-between bg-white px-6 py-3 rounded-lg border">
      <div className="text-sm text-gray-700">
        Page {currentPage} of {totalPages}
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="p-2 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 rounded"
        >
          <ChevronLeftIcon className="h-5 w-5" />
        </button>
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="p-2 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 rounded"
        >
          <ChevronRightIcon className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default Pagination;
