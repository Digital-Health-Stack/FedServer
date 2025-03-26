import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  const gotoRegister = () => {
    navigate("/register");
  };

  return (
    <div className="flex h-screen items-center justify-center flex-col">
      <h1 className="text-3xl font-bold text-center text-blue-500">
        Welcome to FedClient
      </h1>
      <button
        onClick={gotoRegister}
        className="mt-4 px-4 py-2 text-white bg-blue-500 rounded-md"
      >
        Register
      </button>
    </div>
  );
}
