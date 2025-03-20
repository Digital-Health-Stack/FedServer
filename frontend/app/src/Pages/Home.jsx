import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  const gotoRegister = () => {
    navigate("/register");
  };

  return (
    <div className="flex flex-col items-center">
      <h1 className="text-3xl font-bold underline text-center text-red-500">
        Click on the Vite and React logos to learn more code
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
