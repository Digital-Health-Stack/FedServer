// Component for displaying label-value pairs
const InfoItem = ({ label, value }) => (
  <div className="flex flex-col">
    <span className="text-sm font-medium text-gray-500">{label}</span>
    <span className="text-lg font-semibold bg-gray-50 border border-gray-300 rounded-lg p-2 min-h-[40px]">
      {value ?? "N/A"}
    </span>
  </div>
);

export default InfoItem;
