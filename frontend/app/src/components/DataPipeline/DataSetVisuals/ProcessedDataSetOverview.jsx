// import React, { useEffect, useState } from "react";
// import axios from "axios";
// import { useParams } from "react-router-dom";
// import SummaryStats from "./DatasetDetails/SummaryStats";
// import ColumnDetails from "./DatasetDetails/ColumnDetails";
// import Tasks from "./DatasetDetails/TaskCard";
// import PreprocessingDetails from "./DatasetDetails/PreprocessingDetails";

// const PROCESSED_DATASET_URL = process.env.REACT_APP_PROCESSED_OVERVIEW_PATH;

// const ProcessedDataSetOverview = () => {
//   const { filename } = useParams();
//   const [dataset, setDataset] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     const fetchDataset = async () => {
//       try {
//         const response = await axios.get(
//           `${PROCESSED_DATASET_URL}/${filename}`
//         );
//         setDataset(response.data);
//       } catch (err) {
//         setError(err.message);
//       } finally {
//         setLoading(false);
//       }
//     };
//     fetchDataset();
//   }, [filename]);

//   if (loading) return <div className="p-4">Loading dataset...</div>;
//   if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
//   if (!dataset) return <div className="p-4">Dataset not found</div>;

//   const columnDetails = {};
//   dataset.datastats.columnStats.forEach((column) => {
//     columnDetails[column.name] = column.type;
//   });

//   return (
//     <div className="space-y-6">
//       <SummaryStats
//         filename={filename}
//         numRows={dataset.datastats?.numRows}
//         numCols={dataset.datastats?.numColumns}
//       />

//       {dataset.datastats?.columnStats && (
//         <ColumnDetails columnStats={dataset.datastats.columnStats} />
//       )}

//       <Tasks datasetId={dataset.dataset_id} />

//       <PreprocessingDetails
//         columns={columnDetails}
//         filename={filename}
//         directory="processed"
//       />
//     </div>
//   );
// };

// export default ProcessedDataSetOverview;

import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import {
  DocumentTextIcon,
  TableCellsIcon,
  ListBulletIcon,
  WrenchIcon,
} from "@heroicons/react/24/outline";
import DatasetLayout from "./DatasetLayout";
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

  const sections = [
    {
      id: "summary",
      title: "Dataset Summary",
      icon: <DocumentTextIcon className="w-5 h-5" />,
    },
    {
      id: "columns",
      title: "Column Details",
      icon: <TableCellsIcon className="w-5 h-5" />,
    },
    {
      id: "tasks",
      title: "Associated Tasks",
      icon: <ListBulletIcon className="w-5 h-5" />,
    },
    {
      id: "preprocessing",
      title: "Preprocessing",
      icon: <WrenchIcon className="w-5 h-5" />,
    },
  ];

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
    <DatasetLayout sections={sections}>
      <section id="summary" className="scroll-mt-20">
        <SummaryStats
          filename={filename}
          numRows={dataset.datastats?.numRows}
          numCols={dataset.datastats?.numColumns}
        />
      </section>

      <section id="columns" className="scroll-mt-20 mt-12">
        {dataset.datastats?.columnStats && (
          <ColumnDetails columnStats={dataset.datastats.columnStats} />
        )}
      </section>

      <section id="tasks" className="scroll-mt-20 mt-12">
        <Tasks datasetId={dataset.dataset_id} />
      </section>

      <section id="preprocessing" className="scroll-mt-20 mt-12">
        <PreprocessingDetails
          columns={columnDetails}
          filename={filename}
          directory="processed"
        />
      </section>
    </DatasetLayout>
  );
};

export default ProcessedDataSetOverview;
