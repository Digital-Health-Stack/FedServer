import { createContext, useContext, useState } from "react";

const GlobalContext = createContext();

export const useGlobalData = () => useContext(GlobalContext);

export default function MyDataProvider({ children }) {
  const Global_data = {
    Client: { ClientID: "", ClientName: "", DataPath: "", Password: "" },
  };

  const [GlobalData, setGlobalData] = useState(Global_data);

  return (
    <GlobalContext.Provider value={{ GlobalData, setGlobalData }}>
      {children}
    </GlobalContext.Provider>
  );
}
