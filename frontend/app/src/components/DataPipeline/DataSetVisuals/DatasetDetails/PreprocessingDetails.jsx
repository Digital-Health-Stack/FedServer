import React, { useState } from "react";
import axios from "axios";
import { TrashIcon, ArrowUpOnSquareIcon } from "@heroicons/react/24/solid";
import PreprocessingOptions from "../ProcessingComponents/PreprocessingOptions.jsx";

const REACT_APP_PREPROCESS_DATASET_URL =
  process.env.REACT_APP_PREPROCESS_DATASET_URL;

const PreprocessingDetails = ({ columns, filename, directory }) => {
  const [selectedColumn, setSelectedColumn] = useState("");
  const [selectedOption, setSelectedOption] = useState("");
  const [operations, setOperations] = useState([]);
  const [isBannerFixed, setIsBannerFixed] = useState(false);

  const handleOptionChange = (option) => {
    setSelectedOption(option);
  };

  const handleAddSelection = () => {
    if (!selectedOption) return;
    const newOperation = {
      column: selectedColumn || "All Columns",
      operation: selectedOption,
    };
    setOperations([...operations, newOperation]);
    setSelectedOption("");
  };

  const handleRemoveSelection = (index) => {
    const updatedOperations = [...operations];
    updatedOperations.splice(index, 1);
    setOperations(updatedOperations);
  };

  const handleSubmit = async () => {
    const payload = {
      filename: filename,
      directory: directory,
      operations: operations,
    };

    try {
      axios.post(REACT_APP_PREPROCESS_DATASET_URL, payload);
      console.log("Data submitted for preprocessing:", payload);
    } catch (error) {
      console.error("Error in submitting data for preprocessing:", error);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Data Preprocessing</h1>

      {/* Selected Operations */}
      <div className="my-4">
        <h2 className="text-xl font-bold mb-2">Selected Operations:</h2>
        <ul className="space-y-2">
          {operations.map((config, index) => (
            <li
              key={index}
              className="flex items-center justify-between bg-gray-100 p-3 rounded shadow"
            >
              <span>
                Column: <b>{config.column}</b>, Operation:{" "}
                <b>{config.operation}</b>
              </span>
              <button onClick={() => handleRemoveSelection(index)}>
                <TrashIcon className="h-6 w-6 text-red-500 hover:text-red-700" />
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Banner */}
      <div
        className={`${
          isBannerFixed ? "fixed top-0 left-0" : "relative"
        } w-full bg-gray-100 shadow-md p-4 flex flex-wrap gap-4 items-center z-50`}
      >
        {/* Column Selector */}
        <div className="flex-1">
          <label
            htmlFor="column-select"
            className="block text-sm font-medium mb-2"
          >
            Select Column:
          </label>
          <select
            id="column-select"
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            className="p-2 border rounded w-full"
          >
            <option key="all_columns_placeholder" value="">
              All Columns
            </option>
            {Object.keys(columns).map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        {/* Preprocessing Options */}
        <div className="flex-1">
          <label
            htmlFor="column-select"
            className="block text-sm font-medium mb-2"
          >
            Select Operation:
          </label>
          <PreprocessingOptions
            columnName={selectedColumn}
            columnType={
              selectedColumn === "" ? "all_columns" : columns[selectedColumn]
            }
            handleOptionChange={handleOptionChange}
          />
        </div>

        {/* Add and Submit Buttons */}
        <div className="flex-1 flex flex-col md:flex-row items-center gap-5 mt-6">
          <button
            onClick={handleAddSelection}
            className="bg-green-500 text-white px-3 py-1 rounded hover:bg-green-700"
          >
            Add
          </button>
          <button
            onClick={(e) => {
              handleSubmit();
              //   disable button after click
              //   e.target.disabled = true;
              //  hide button after click
              //   e.target.classList.add("hidden");
            }}
            className="bg-indigo-500 text-white px-4 py-1 rounded hover:bg-indigo-700"
          >
            Submit
          </button>
        </div>

        {/* Fix/Unfix Button */}
        <button
          className="absolute top-2 right-2 bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          onClick={() => setIsBannerFixed(!isBannerFixed)}
        >
          <ArrowUpOnSquareIcon className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default PreprocessingDetails;
