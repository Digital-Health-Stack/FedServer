import { BrowserRouter, Route, Router, Routes } from "react-router-dom";
import Home from "./Pages/Home";
import TrainingResults from "./Pages/Results";
import ResultDetails from "./Pages/ResultDetails";
import About from "./Pages/About";
import Error from "./Pages/Error";
import NavBar from "./components/OnWholeApp/NavBar";
import 'bootstrap/dist/css/bootstrap.css';

/*
The App component is the main component of the application. It is the parent component of all the other components.
It contains the NavBar component, which is a navigation bar that allows the user to navigate between different 
pages of the application. The App component also contains the Routes component, which is used to define the
routes of the application. Each route is associated with a specific page component, which is rendered when
the user navigates to that route.
*/

export default function App() {
  return (
    <>
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
    </>
  );
}
