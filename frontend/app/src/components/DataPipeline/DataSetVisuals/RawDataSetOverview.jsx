import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import SummaryStats from "./DatasetDetails/SummaryStats.jsx";
import ColumnDetails from "./DatasetDetails/ColumnDetails.jsx";
import PreprocessingDetails from "./DatasetDetails/PreprocessingDetails.jsx";

const REACT_APP_DATASET_DETAILS_URL = process.env.REACT_APP_DATASET_DETAILS_URL;

const DataSetOverview = () => {
  const [data, setData] = useState(null);
  const dataset_id = useParams().filename;
  const directory = useParams().dir;

  useEffect(() => {
    const loadData = async () => {
      const overview = await axios.get(
        `${REACT_APP_DATASET_DETAILS_URL}/${directory}/${dataset_id}`
      );
      setData(overview.data);
      console.log("file overview data received:", overview.data);
    };

    loadData();
  }, []);

  if (!data) return <p>Loading...</p>;
  if (data.error) return <p>{data.error}</p>;

  const columnDetails = {};
  data.columnStats.forEach((column) => {
    columnDetails[column.name] = column.type;
  });

  return (
    <div>
      <SummaryStats
        fileName={data.fileName}
        numRows={data.numRows}
        numCols={data.numColumns}
      />
      <ColumnDetails columnStats={data.columnStats} />
      <PreprocessingDetails
        columns={columnDetails}
        fileName={data.fileName}
        directory={directory}
      />
    </div>
  );
};

export default DataSetOverview;
