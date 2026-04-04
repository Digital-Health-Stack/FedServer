import React, { useState } from "react";
import { useParams } from "react-router-dom";
import { useEffect } from "react";
import axios from "axios";
import MetricsChart from "../components/ResultDetails/MetricsChart";
import "bootstrap/dist/css/bootstrap.css";
import "bootstrap/dist/js/bootstrap.js";

const get_training_detail_from_session_id =
  "http://localhost:8000/get-training-result";

const RenderData = ({ data, level = 0 }) => {
  if (typeof data === "object" && data !== null) {
    if (Array.isArray(data)) {
      return (
        <ul className="list-group ms-3">
          {data.map((item, index) => (
            <li className="list-group-item" key={index}>
              <RenderData data={item} level={level + 1} />
            </li>
          ))}
        </ul>
      );
    } else {
      return (
        <div className="ms-3">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="mb-2">
              <strong>{capitalizeFirstLetter(key)}:</strong>
              <RenderData data={value} level={level + 1} />
            </div>
          ))}
        </div>
      );
    }
  } else {
    return (
      <div className="card ms-3 mb-2" style={{ display: "inline-block" }}>
        <div className="card-body p-2">{data?.toString()}</div>
      </div>
    );
  }
};
const capitalizeFirstLetter = (string) => {
  return string.charAt(0).toUpperCase() + string.slice(1);
};

const Results = ({ data }) => {
  const testResults = data.test_results;
  const [showDetails, setShowDetails] = useState(false);

  const { organisation_name, model_name, model_info, dataset_info } =
    data.session_data;
  const toggleDetails = () => {
    setShowDetails(!showDetails);
  };

  return (
    <div className="container mt-4">
      {/* Show organization name, model name, and a button to toggle details */}
      <div className="mb-4 d-flex">
        <div className="d-flex align-items-center me-4">
          {" "}
          {/* Add margin-end to create gap */}
          <div className="form-group d-flex flex-column me-3">
            <label htmlFor="organisationName" className="form-label mb-1">
              Organization Name:
            </label>
            <input
              type="text"
              id="organisationName"
              className="form-control"
              value={organisation_name}
              readOnly
              style={{ maxWidth: "300px" }} // Adjust width as needed
            />
          </div>
        </div>
        <div className="d-flex align-items-center">
          <div className="form-group d-flex flex-column">
            <label htmlFor="modelName" className="form-label mb-1">
              Model Name:
            </label>
            <input
              type="text"
              id="modelName"
              className="form-control"
              value={model_name}
              readOnly
              style={{ maxWidth: "300px" }} // Adjust width as needed
            />
          </div>
        </div>
      </div>

      {/* Button to toggle details */}
      <div className="text-center mb-4">
        <button className="btn btn-success" onClick={toggleDetails}>
          {showDetails ? "Hide Details" : "Show Details"}
        </button>
      </div>

      <div className="card">
        <div className="card-header text-center bg-dark text-white">
          <h3>Round Results</h3>
        </div>
        <div className="card-body">
          <table className="table table-bordered">
            <thead>
              <tr className="table-secondary">
                <th>Round</th>
                {Object.keys(testResults["round 0"]).map((metric) => (
                  <th key={metric}>{capitalizeFirstLetter(metric)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(testResults).map(([round, result]) => (
                <tr key={round}>
                  <td>{capitalizeFirstLetter(round)}</td>
                  {Object.values(result).map((value, index) => (
                    <td key={index}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Render the MetricsChart component */}
      <MetricsChart testResults={testResults} />

      {/* Conditionally render model_info and dataset_info */}
      {showDetails && (
        <div className="card mt-4">
          <div className="card-body">
            <h4 className="text-primary border-bottom pb-2">
              Model Information
            </h4>
            <RenderData data={model_info} />

            <h4 className="text-primary border-bottom pb-2 mt-4">
              Dataset Information
            </h4>
            <RenderData data={dataset_info} />
          </div>
        </div>
      )}
    </div>
  );
};

export default function ResultDetails() {
  const { sessionId } = useParams();
  const [resultDetails, setresultDetails] = useState({});

  const fetchResultDetails = async () => {
    const get_result_detail_endpoint = `${get_training_detail_from_session_id}/${sessionId}`;
    console.log("getting result from: ", get_result_detail_endpoint);
    try {
      const res = await axios.get(get_result_detail_endpoint);
      console.log("result detail fetched from server:", res.data);
      setresultDetails(res.data);
    } catch (error) {
      console.log("Error Fetching Data", error);
    }
  };

  useEffect(() => {
    fetchResultDetails();
  }, []);

  return Object.keys(resultDetails).length === 0 ? (
    <div className="alert alert-warning text-center mt-4" role="alert">
      SessionID Does Not exist!!
    </div>
  ) : (
    <div className="container-md">
      <div className="card bg-light">
        <div className="card-header text-center bg-dark text-white">
          <h3>Training Details</h3>
        </div>
        <Results data={resultDetails} />
      </div>
    </div>
  );
}
