import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center bg-blue-50 text-red-900 h-screen overflow-hidden">
      <h1 className="text-6xl font-bold text-red-500">404</h1>
      <h2 className="text-2xl font-semibold mt-2">Page Not Found</h2>
      <p className="mt-2 text-gray-400">
        Oops! The page you are looking for does not exist.
      </p>

      <button
        onClick={() => navigate("/")}
        className="mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg transition-all duration-300"
      >
        Go Home
      </button>
    </div>
  );
}
