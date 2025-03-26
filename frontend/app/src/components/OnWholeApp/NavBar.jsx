import React, { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Bars3Icon,
  XMarkIcon,
  HomeIcon,
  UserIcon,
  ChartBarIcon,
  ServerStackIcon,
  ClipboardDocumentListIcon,
  TrophyIcon,
} from "@heroicons/react/24/solid";

const NavBar = () => {
  const [isNavbarOpen, setIsNavbarOpen] = useState(false);

  const handleToggle = () => {
    setIsNavbarOpen(!isNavbarOpen);
  };

  return (
    <nav className="bg-gray-900 border-b border-gray-700 text-white">
      <div className="container mx-auto flex justify-between items-center p-1/2">
        {/* Logo */}
        <a className="text-xl font-bold flex items-center text-white" href="/">
          <span>FedServer</span>
        </a>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-white focus:outline-none"
          onClick={handleToggle}
          aria-expanded={isNavbarOpen}
          aria-label="Toggle navigation"
        >
          {isNavbarOpen ? (
            <XMarkIcon className="w-7 h-7" />
          ) : (
            <Bars3Icon className="w-7 h-7" />
          )}
        </button>

        {/* Navigation Links - Now Right Aligned */}
        <div
          className={`absolute md:static top-10 right-0 w-full md:w-auto bg-gray-900 md:bg-transparent md:flex md:items-center p-4 md:p-0 transition-all duration-300 ${
            isNavbarOpen ? "block" : "hidden"
          } md:ml-auto`}
        >
          <ul className="md:flex justify-end items-center space-y-4 md:space-y-0 md:space-x-6 w-full">
            <li>
              <NavLink
                className="flex items-center gap-2 py-2 px-4 hover:text-gray-400"
                to="/"
              >
                <HomeIcon className="w-5 h-5" /> Home
              </NavLink>
            </li>
            <li>
              <NavLink
                className="flex items-center gap-2 py-2 px-4 hover:text-gray-400"
                to="/Register"
              >
                <UserIcon className="w-5 h-5" /> Register
              </NavLink>
            </li>
            <li>
              <NavLink
                className="flex items-center gap-2 py-2 px-4 hover:text-gray-400"
                to="/trainings"
              >
                <ChartBarIcon className="w-5 h-5" /> Trainings
              </NavLink>
            </li>
            <li>
              <NavLink
                className="flex items-center gap-2 py-2 px-4 hover:text-gray-400"
                to="/data-quality"
              >
                <ServerStackIcon className="w-5 h-5" /> Access Data Quality
              </NavLink>
            </li>
            <li>
              <NavLink
                className="flex items-center gap-2 py-2 px-4 hover:text-gray-400"
                to="/ManageData"
              >
                <ClipboardDocumentListIcon className="w-5 h-5" /> Datasets
              </NavLink>
            </li>
            <li>
              <NavLink
                className="flex items-center gap-2 py-2 px-4 hover:text-gray-400"
                to="/benchmarks"
              >
                <TrophyIcon className="w-5 h-5" /> Benchmarks
              </NavLink>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
