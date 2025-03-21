import "./index.css";

import Home from "./Pages/Home";
import Error from "./Pages/Error";
import NavBar from "./components/OnWholeApp/NavBar";

import { Route, Routes } from "react-router-dom";

export default function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" exact element={<Home />} />
        <Route path="*" element={<Error />} />
      </Routes>
    </>
  );
}
