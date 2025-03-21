import "./index.css";

import Home from "./Pages/Home";
import Error from "./Pages/Error";
import NavBar from "./components/OnWholeApp/NavBar";

import { Route, Routes } from "react-router-dom";

export default function App() {
  return (
    <>
<<<<<<< HEAD
        <NavBar />
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            margin: "100px",
          }}
        >
          <Routes>
            <Route path="/" exact element={<Home />} />

            <Route path="/Results" element={<TrainingResults />} />

            <Route
              path="/TrainingResults/details/:sessionId"
              element={<ResultDetails />}
            />

            <Route path="/About" element={<About />} />

            <Route path="/*" element={<Error />} />
          </Routes>
        </div>
=======
      <NavBar />
      <Routes>
        <Route path="/" exact element={<Home />} />
        <Route path="*" element={<Error />} />
      </Routes>
>>>>>>> next-version
    </>
  );
}
