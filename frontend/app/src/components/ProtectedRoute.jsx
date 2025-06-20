import { useLocation, Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export const PrivateRoute = ({ children }) => {
  const { user } = useAuth();
  const location = useLocation();

  return user ? (
    children
  ) : (
    <Navigate to="/admin-login" state={{ from: location }} replace />
  );
};

export const OnlyGuestRoute = ({ children }) => {
  const { user } = useAuth();

  return user ? <Navigate to={"/"} /> : children;
};
