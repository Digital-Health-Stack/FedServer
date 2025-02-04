import React from "react";
import FedClientImage from "../assets/FedClient.jpeg";
import StepsGif from "../assets/steps.gif"; // in backgrd
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  const gotoRegister = () => {
    navigate("/register");
  };

  return (
    <div className="container-fluid d-flex align-items-center">
      <div className="row w-100">
        <div className="col-md-5 d-none d-md-block">
          <img src={StepsGif} alt="Steps" className="img-fluid" />
        </div>
        <div className="col-md-7 d-flex justify-content-center align-items-center">
          <div className="card shadow-md mb-5 rounded">
            <div className="card-body">
              <div className="text-center mb-4">
                <img
                  src={FedClientImage}
                  className="rounded-circle"
                  alt="FedClient"
                  width="150"
                  height="150"
                />
              </div>
              <h1 className="text-center mb-4 text-dark">
                Welcome to FedClient
              </h1>
              <div className="alert alert-info text-center" role="alert">
                <strong>
                  This is Client Application to simulate Federated Learning
                </strong>
              </div>
              <button className="btn btn-success w-100" onClick={gotoRegister}>
                Get Started
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
