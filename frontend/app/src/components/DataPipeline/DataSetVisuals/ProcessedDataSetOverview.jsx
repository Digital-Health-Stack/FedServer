import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import SummaryStats from "./DatasetDetails/SummaryStats";
import ColumnDetails from "./DatasetDetails/ColumnDetails";
import Tasks from "./DatasetDetails/TaskCard";
import PreprocessingDetails from "./DatasetDetails/PreprocessingDetails";

const PROCESSED_DATASET_URL = process.env.REACT_APP_PROCESSED_OVERVIEW_PATH;

const ProcessedDataSetOverview = () => {
  const { filename } = useParams();
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const response = await axios.get(
          `${PROCESSED_DATASET_URL}/${filename}`
        );
        setDataset(response.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, [filename]);

  if (loading) return <div className="p-4">Loading dataset...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!dataset) return <div className="p-4">Dataset not found</div>;

  const columnDetails = {};
  dataset.datastats.columnStats.forEach((column) => {
    columnDetails[column.name] = column.type;
  });

  return (
    <div className="space-y-6">
      <SummaryStats
        filename={filename}
        numRows={dataset.datastats?.numRows}
        numCols={dataset.datastats?.numColumns}
      />

      {dataset.datastats?.columnStats && (
        <ColumnDetails columnStats={dataset.datastats.columnStats} />
      )}

      <Tasks datasetId={dataset.dataset_id} />

      <PreprocessingDetails
        columns={columnDetails}
        filename={filename}
        directory="processed"
      />
    </div>
  );
};

export default ProcessedDataSetOverview;
