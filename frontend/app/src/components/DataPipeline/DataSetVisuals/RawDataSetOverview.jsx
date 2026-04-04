import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import SummaryStats from "./DatasetDetails/SummaryStats.jsx";
import ColumnDetails from "./DatasetDetails/ColumnDetails.jsx";
import PreprocessingDetails from "./DatasetDetails/PreprocessingDetails.jsx";
import { getRawDatasetDetail } from "../../../services/privateService.js";

// const RAW_DATASET_DETAILS_URL = process.env.REACT_APP_RAW_OVERVIEW_PATH;

const DataSetOverview = () => {
  const [data, setData] = useState(null);
  const filename = useParams().filename;

  useEffect(() => {
    const loadData = async () => {
      const overview = await getRawDatasetDetail(filename);
      setData(overview.data);
      console.log("file overview data received:", overview.data);
    };

    loadData();
  }, []);

  if (!data) return <p>Loading...</p>;
  if (data.error) return <p>{data.error}</p>;

  const columnDetails = {};
  if (data.datastats && data.datastats.columnStats) {
    data.datastats.columnStats.forEach((column) => {
      columnDetails[column.name] = column.type;
    });
  }

  return (
    <div>
      <SummaryStats
        filename={filename}
        description={data.description}
        numRows={data.datastats?.numRows}
        numCols={data.datastats?.numColumns}
      />
      <ColumnDetails columnStats={data.datastats?.columnStats || []} />
      <PreprocessingDetails
        columns={columnDetails}
        filename={filename}
        directory="raw"
      />
    </div>
  );
};

export default DataSetOverview;
