import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { useState } from "react";
import Card from "../components/Card/Card";

const getAllTrainingResultsURL =
  "http://localhost:8000/get-all-completed-trainings";

export default function TrainingResults() {
  const navigate = useNavigate();
  const [completedTrainings, setcompletedTrainings] = useState([]);

  const opendetails = (item) => {
    navigate(`/TrainingResults/details/${item}`);
  };

  useEffect(() => {
    async function fetchTrainingData(url) {
      try {
        console.log("fetching training from: ", url);
        const res = await axios.get(url);
        if (res.status == 200) {
          const completedTrainings = res.data["results"];
          setcompletedTrainings(completedTrainings);
        } else {
          console.log(
            "Failed to fetch completed trainings data from server, status: ",
            res.status,
          );
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    }
    console.log("fetching completed trainings..");
    fetchTrainingData(getAllTrainingResultsURL);
  }, []);

  return (
    <div className="container">
      {completedTrainings.length === 0 ? (
        <div className="alert alert-warning text-center mt-4" role="alert">
          No Completed Trainings Available!!
        </div>
      ) : (
        <div className="row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4">
          {completedTrainings.map((item) => (
            <Card key={item.session_id} item={item} opendetails={opendetails} />
          ))}
        </div>
      )}
    </div>
  );
}
