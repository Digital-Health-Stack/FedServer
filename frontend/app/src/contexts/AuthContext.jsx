import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useDebugValue,
} from "react";

import {
  refreshAccessToken,
  logout as logsTheOut,
} from "../services/authService";

import axios from "axios";
import { BASE_URL } from "../services/config";

// import { toast } from 'react-toastify';
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const AuthContext = createContext();

const REFRESH_INTERVAL = 1000 * 60 * 15; // 15 minutes

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [eventSource, setEventSource] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  let refreshTimeout;

  useEffect(() => {
    const savedUser = JSON.parse(localStorage.getItem("user"));
    if (savedUser) {
      setUser(savedUser);
      scheduleTokenRefresh();
    }
    setLoading(false);
  }, []);

  // useEffect(() => {
  //     const initializeEventSource = async () => {
  //         if (user && user.access_token) {
  //             const user = JSON.parse(localStorage.getItem("user"));
  //             const headers = {};

  //             if (user && user.access_token) {
  //                 headers["Authorization"] = `Bearer ${user.access_token}`;
  //             }

  //             const response = await fetch(`${BASE_URL}/notifications/stream`, {
  //                 method: "GET",
  //                 headers: headers,
  //             });

  //             const reader = response.body.getReader();
  //             const decoder = new TextDecoder("utf-8");

  //             let buffer = "";

  //             while (true) {
  //                 const { value, done } = await reader.read();
  //                 if (done) {
  //                     console.log("Stream closed.");
  //                     break;
  //                 }

  //                 buffer += decoder.decode(value, { stream: true });

  //                 // Process SSE messages
  //                 const lines = buffer.split("\n");
  //                 for (const line of lines) {
  //                     if (line.startsWith("data:")) {
  //                         const jsonData = line.replace("data:", "").trim();
  //                         try {
  //                             const notifications = JSON.parse(jsonData);

  //                             for (let notification of notifications) {
  //                                 handleNotification(notification)
  //                             }
  //                             console.log("Received data:", notifications);
  //                         }
  //                         catch (exception) {
  //                             console.log("Received data:", exception, jsonData)
  //                         }
  //                     }
  //                 }

  //                 // Keep only unprocessed data
  //                 buffer = buffer.slice(buffer.lastIndexOf("\n") + 1);
  //             }
  //         }
  //     };

  //     // Initialize the connection
  //     initializeEventSource();

  //     return () => {
  //         if (eventSource) {
  //             eventSource.close();
  //         }
  //     };
  // }, [user]);

  const login = (userData) => {
    setUser(userData);

    localStorage.setItem("user", JSON.stringify(userData));

    scheduleTokenRefresh();
  };

  const logout = () => {
    logsTheOut(api).catch(console.error).finally(removeUserData);
  };

  const removeUserData = () => {
    setUser(null);
    localStorage.removeItem("user");
    clearTimeout(refreshTimeout);
  };

  const scheduleTokenRefresh = () => {
    // Clear existing refresh to avoid duplicate timers
    clearTimeout(refreshTimeout);

    refreshTimeout = setTimeout(updateTokens, REFRESH_INTERVAL);
  };

  const updateTokens = () => {
    const userData = JSON.parse(localStorage.getItem("user"));
    if (userData && userData.refresh_token)
      refreshAccessToken(api, userData.refresh_token)
        .then((response) => {
          setUser(response.data);
          localStorage.setItem("user", JSON.stringify(response.data));
          scheduleTokenRefresh();
        })
        .catch((error) => {
          console.error("Failed to refresh access token:", error);
          logout();
        });
  };

  // const handleNotification = ({ type, message, data, session_id }) => {
  //     console.log(type)
  //     if (type == 'new-session') {
  //         // if (location.pathname === '/TrainingStatus') {
  //         //     // window.location.reload();
  //         //     navigate('/TrainingStatus')
  //         // }

  //         toast.info(message, {
  //             onClick: () => {
  //                 navigate(`/TrainingStatus/details/${session_id}`)
  //             }
  //         })
  //     }
  //     else if (type == "get_model_parameters_start_background_process") {
  //         console.log("Config before initialising: ", data, session_id);

  //         console.log("building model on client side...");
  //         setUpModel(data, user, session_id, api); // Function to initialize training
  //     }
  //     else if (type == "start_training"){
  //         console.log("start training on client side...");
  //         trainModel(data["local_model_id"]);
  //     }
  // }

  const api = axios.create({
    baseURL: BASE_URL,
    timeout: 50000,
    headers: {
      "Content-Type": "application/json",
      // 'Authorization': `Bearer ${user ? user.access_token : ''}` // Add the token
    },
  });

  // Add a request interceptor to add the token to the request
  api.interceptors.request.use(
    (config) => {
      //   const token = localStorage.getItem('token'); // Retrieve token from local storage
      const user = JSON.parse(localStorage.getItem("user"));
      if (user) {
        config.headers["Authorization"] = `Bearer ${user.access_token ?? ""}`;
      }
      return config;
    },
    (error) => Promise.reject(error),
  );

  api.interceptors.response.use(
    (response) => {
      // Any status code within the range of 2xx will trigger this function
      return response;
    },
    (error) => {
      // Any status codes outside the range of 2xx will trigger this function
      if (error.response) {
        // Server responded with a status code outside the range of 2xx
        console.error("Error response:", error.response);

        // Handle specific status codes as needed
        if (error.response.status === 401) {
          // alert("Unauthorized access - please log in again.");
          toast.error("Unauthorized access - please log in again.", {
            position: "bottom-center",
            autoClose: 3000,
          });
          removeUserData();
          // Optionally, log the user out, redirect to login, or refresh the token
        } else if (error.response.status === 403) {
          // alert("Access denied - you do not have permission.");
          toast.error("Access denied - you do not have permission.", {
            position: "bottom-center",
            autoClose: 3000,
          });
        } else if (error.response.status === 500) {
          // alert("Internal server error - please try again later.");
          toast.error("Internal server error - please try again later.", {
            position: "bottom-center",
            autoClose: 3000,
          });
        }
      } else if (error.request) {
        // No response was received from the server
        console.error("Error request:", error.request);
        // alert("Network error - please check your internet connection.");
        toast.error("Network error - please check your internet connection.", {
          position: "bottom-center",
          autoClose: 3000,
        });
      } else {
        // Error occurred in setting up the request
        console.error("Error message:", error.message);
        // alert("An error occurred - please try again.");
        toast.error("An error occurred - please try again.", {
          position: "bottom-center",
          autoClose: 3000,
        });
      }
      return Promise.reject(error); // Optionally, you can also return a custom error message here
    },
  );

  // const sseApi = axios.create({
  //     baseURL: BASE_URL,
  //     responseType: "stream",
  //     timeout: 2 * 24 * 60 * 60 * 1000
  // });

  // sseApi.interceptors.request.use(
  //     (config) => {
  //         //   const token = localStorage.getItem('token'); // Retrieve token from local storage
  //         const user = JSON.parse(localStorage.getItem('user'));
  //         if (user) {
  //             config.headers['Authorization'] = `Bearer ${user.access_token ?? ''}`;
  //         }
  //         return config;
  //     },
  //     (error) => Promise.reject(error)
  // );

  return (
    <AuthContext.Provider value={{ user, login, logout, api }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);

// const setUpModel = async (config, user, sessionId, api) => {
//     console.log("config received by setUpModel: ", config);
//     const data = {
//         model_config: config,
//         session_id: sessionId,
//         client_id: user.access_token,
//     };

//     initializeModel(data)
//         .then(({ data }) => {
//             console.log(data.message)

//             const status_four_data = {
//                 session_id: sessionId,
//                 decision: 1,
//                 local_model_id: data.local_model_id
//             };

//             sendModelInitiation(api, status_four_data)
//                 .then(({ data: { message } }) => console.log(message))
//                 .catch(console.error)
//         })
// };

// const trainModel = (local_model_id) => {
//     trainModelService(local_model_id)
//         .then(({ data }) => {
//             if (data && data.status === 200) {
//                 console.log("output:", data.stdout);
//                 console.log("stderr:", data.stderr);
//                 console.log("returncode:", data.returncode);
//             } else {
//                 console.error("Failed to start the execution on the private server", data);
//             }
//         })
//         .catch((error) => {
//             console.error("Error in trainModelService:", error);
//         });
// };
