import "./index.css";
import { Route, Routes } from "react-router-dom";
import Home from "./Pages/Home";
import Register from "./Pages/Register";
import About from "./Pages/About";
import Error from "./Pages/Error";
import NavBar from "./components/OnWholeApp/NavBar";
import MyDataProvider from "./GlobalContext";
// import EventsAction from "./EventsActions";
import { useState } from "react";
import { AuthProvider } from "./contexts/AuthContext";
import { PrivateRoute, OnlyGuestRoute } from "./components/ProtectedRoute";
import { ToastContainer } from "react-toastify";
// import "react-toastify/dist/ReactToastify.min.css";
import ManageData from "./Pages/ManageData";
import ViewRecentUploads from "./components/DataPipeline/ViewRecentUploads";
import ViewAllDatasets from "./components/DataPipeline/ViewAllDatasets";
import RawDataSetOverview from "./components/DataPipeline/DataSetVisuals/RawDataSetOverview.jsx";
import ProcessedDataSetOverview from "./components/DataPipeline/DataSetVisuals/ProcessedDataSetOverview.jsx";
import PreprocessingDocs from "./components/DataPipeline/DataSetVisuals/ProcessingComponents/PreprocessingDocs.jsx";

export default function App() {
  const [clientToken, setClientToken] = useState("");
  const [sessions, setSessions] = useState([]);
  const [socket, setSocket] = useState(null);
  return (
    <>
      <MyDataProvider>
        <AuthProvider>
          <ToastContainer />
          {/* <EventsAction socket={socket} clientToken={clientToken} /> */}
          <NavBar />
          <Routes>
            <Route path="/" exact element={<Home />} />

            <Route
              path="/Register"
              element={
                <OnlyGuestRoute>
                  <Register
                    clientToken={clientToken}
                    setClientToken={setClientToken}
                    setSocket={setSocket}
                  />
                </OnlyGuestRoute>
              }
            />

            <Route path="/About" element={<About />} />
            <Route path="/ManageData" element={<ManageData />} />
            <Route
              path="/view-recent-uploads"
              element={
                <PrivateRoute>
                  <ViewRecentUploads />
                </PrivateRoute>
              }
            />
            <Route path="/view-all-datasets" element={<ViewAllDatasets />} />
            <Route
              path="/raw-dataset-overview/:filename"
              element={<RawDataSetOverview />}
            />
            <Route
              path="/processed-dataset-overview/:filename"
              element={<ProcessedDataSetOverview />}
            />
            <Route path="/preprocessing-docs" element={<PreprocessingDocs />} />
            <Route path="/*" element={<Error />} />
          </Routes>
        </AuthProvider>
      </MyDataProvider>
    </>
  );
}
