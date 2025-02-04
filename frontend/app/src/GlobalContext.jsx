import { createContext, useContext, useState } from "react";

const GlobalContext = createContext();

export const useGlobalData = () => useContext(GlobalContext);

export default function MyDataProvider({ children }) {
  const Global_data = {
    Client: { ClientID: "", ClientName: "", DataPath: "", Password: "" },
    ConnectionObject: null,
    CurrentModels: [
      {
        RequestId: "123",
        OrgName: "Sample",
        Status: "Unknown",
        Model: {
          MLP: {
            input_layer: {
              activation_function: "relu",
              input_shape: "(784,)",
              num_nodes: "784",
            },
            intermediate_layer: {
              0: { feature_name: "200", activation_function: "relu" },
              1: { feature_name: "20", activation_function: "softmax" },
            },
            loss_function: "mae",
            optimizer: "sgd",
            output_layer: { activation_function: "leakyRelu", num_nodes: "10" },
          },
        },
        Data: {
          about_dataset: "dv",
          feature_list: {
            0: { feature_name: "age", type_Of_feature: "int" },
            1: { feature_name: "height", type_Of_feature: "int" },
            2: { feature_name: "unknown", type_Of_feature: "bool" },
            3: { feature_name: "gender", type_Of_feature: "bool" },
          },
        },
      },
    ],
  };

  const [GlobalData, setGlobalData] = useState(Global_data);

  return (
    <GlobalContext.Provider value={{ GlobalData, setGlobalData }}>
      {children}
    </GlobalContext.Provider>
  );
}
