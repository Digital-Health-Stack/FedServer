import "./index.css";
import { Route, Routes } from "react-router-dom";
import Dashboard from "./Pages/Dashboard.jsx";
import AdminLogin from "./Pages/AdminLogin";
import About from "./Pages/About";
import Error from "./Pages/Error";
import NavBar from "./components/OnWholeApp/NavBar";
import MyDataProvider from "./GlobalContext";

import { useState } from "react";
import { AuthProvider } from "./contexts/AuthContext";
import { PrivateRoute, OnlyGuestRoute } from "./components/ProtectedRoute";
import { ToastContainer } from "react-toastify";

import ManageData from "./Pages/ManageData";
import AssessDataQuality from "./Pages/AssessDataQuality.jsx";
import ViewRecentUploads from "./components/DataPipeline/ViewRecentUploads";
import ViewAllDatasets from "./components/DataPipeline/ViewAllDatasets";
import ProcessedDataSetOverview from "./components/DataPipeline/DataSetVisuals/ProcessedDataSetOverview.jsx";
import PreprocessingDocs from "./components/DataPipeline/DataSetVisuals/ProcessingComponents/PreprocessingDocs.jsx";
import Benchmarks from "./components/DataPipeline/DataSetVisuals/DatasetDetails/BenchmarkList.jsx";
import BenchmarkTraining from "./components/DataPipeline/DataSetVisuals/DatasetDetails/BenchmarkTraining.jsx";
import Trainings from "./Pages/Trainings.jsx";
import TrainingDetails from "./Pages/TrainingDetails.jsx";
import Leaderboard from "./Pages/Leaderboard.jsx";

export default function App() {
  const [clientToken, setClientToken] = useState("");
  const [sessions, setSessions] = useState([]);
  const [socket, setSocket] = useState(null);
  return (
    <>
      <MyDataProvider>
        <AuthProvider>
          <ToastContainer position="bottom-center" autoClose={3000} />
          {/* <EventsAction socket={socket} clientToken={clientToken} /> */}
          <NavBar />
          <Routes>
            <Route
              path="/"
              exact
              element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              }
            />

            <Route
              path="/admin-login"
              element={
                <OnlyGuestRoute>
                  <AdminLogin
                    clientToken={clientToken}
                    setClientToken={setClientToken}
                    setSocket={setSocket}
                  />
                </OnlyGuestRoute>
              }
            />

            <Route
              path="/About"
              element={
                <PrivateRoute>
                  <About />
                </PrivateRoute>
              }
            />
            <Route
              path="/ManageData"
              element={
                <PrivateRoute>
                  <ManageData />
                </PrivateRoute>
              }
            />
            <Route
              path="/view-recent-uploads"
              element={
                <PrivateRoute>
                  <ViewRecentUploads />
                </PrivateRoute>
              }
            />
            <Route
              path="/view-all-datasets"
              element={
                <PrivateRoute>
                  <ViewAllDatasets />
                </PrivateRoute>
              }
            />
            <Route
              path="/processed-dataset-overview/:filename"
              element={
                <PrivateRoute>
                  <ProcessedDataSetOverview />
                </PrivateRoute>
              }
            />
            <Route
              path="/preprocessing-docs"
              element={
                <PrivateRoute>
                  <PreprocessingDocs />
                </PrivateRoute>
              }
            />

            <Route
              path="/tasks/:task_id/benchmarks"
              element={
                <PrivateRoute>
                  <Benchmarks />
                </PrivateRoute>
              }
            />

            <Route
              path="/history/:task_id"
              element={
                <PrivateRoute>
                  <Leaderboard />
                </PrivateRoute>
              }
            />
            <Route
              path="/assess-data-quality"
              element={
                <PrivateRoute>
                  <AssessDataQuality />
                </PrivateRoute>
              }
            />
            <Route
              path="/trainings"
              element={
                <PrivateRoute>
                  <Trainings />
                </PrivateRoute>
              }
            />
            <Route
              path="/trainings/:sessionId"
              element={
                <PrivateRoute>
                  <TrainingDetails />
                </PrivateRoute>
              }
            />
            <Route path="/*" element={<Error />} />
          </Routes>
        </AuthProvider>
      </MyDataProvider>
    </>
  );
}
