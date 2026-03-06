import React, { createContext, useContext, useState } from "react";

const HelpContext = createContext();

export const useHelp = () => {
  const context = useContext(HelpContext);
  if (!context) {
    throw new Error("useHelp must be used within a HelpProvider");
  }
  return context;
};

export const HelpProvider = ({ children }) => {
  const [showWalkthrough, setShowWalkthrough] = useState(false);

  const startWalkthrough = () => {
    setShowWalkthrough(true);
  };

  const stopWalkthrough = () => {
    setShowWalkthrough(false);
  };

  return (
    <HelpContext.Provider
      value={{ showWalkthrough, startWalkthrough, stopWalkthrough }}
    >
      {children}
    </HelpContext.Provider>
  );
};
