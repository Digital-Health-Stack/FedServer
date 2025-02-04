import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  const gotoHome = () => {
    navigate("/");
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        flexDirection: "column",
      }}
    >
      <h4>Error 404...</h4>
      <div style={{ height: "1rem" }}></div> {/* Blank line for spacing */}
      <p>No such Page Available</p>
      <button onClick={gotoHome}>Home</button>
    </div>
  );
}
