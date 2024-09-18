import React from "react";
import FedprocessGif from "../assets/fedprocess.gif";
import AboutIcon from "../assets/aboutsicon.png";

export default function About() {
  return (
    <div
      style={{
        display: "flex ",
        maxWidth: "100%",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          flex: 1,
          display: "flex",
          margin: "20px",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            display: "flex",
            margin: "20px",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            flex: 1,
          }}
        >
          <img
            src={AboutIcon}
            alt="FedClient"
            style={{
              width: "auto",
              height: "auto",
              maxHeight: "200px",
              objectFit: "contain",
            }}
          />
          <h3 style={{ margin: 0, textAlign: "center" }}>How it Works</h3>
        </div>
      </div>
      <div
        style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <img
          src={FedprocessGif}
          alt="Fedprocess Gif"
          style={{
            width: "auto",
            height: "auto",
            maxHeight: "500px",
            objectFit: "contain",
          }}
        />
      </div>
    </div>
  );
}
