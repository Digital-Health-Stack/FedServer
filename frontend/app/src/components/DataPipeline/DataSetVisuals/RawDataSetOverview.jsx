import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import SummaryStats from "./DatasetDetails/SummaryStats.jsx";
import ColumnDetails from "./DatasetDetails/ColumnDetails.jsx";
import PreprocessingDetails from "./DatasetDetails/PreprocessingDetails.jsx";

const RAW_DATASET_DETAILS_URL = process.env.REACT_APP_RAW_OVERVIEW_PATH;

const DataSetOverview = () => {
  const [data, setData] = useState(null);
  const filename = useParams().filename;

  useEffect(() => {
    const loadData = async () => {
      const overview = await axios.get(
        `${RAW_DATASET_DETAILS_URL}/${filename}`
      );
      setData(overview.data.datastats);
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
        fileName={data.filename}
        numRows={data.numRows}
        numCols={data.numColumns}
      />
      <ColumnDetails columnStats={data.columnStats} />
      <PreprocessingDetails
        columns={columnDetails}
        fileName={data.filename}
        directory="Uploads"
      />
    </div>
  );
};

export default DataSetOverview;
