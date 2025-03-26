import { Route, Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export const PrivateRoute = ({ children }) => {
    const { user } = useAuth(); // Access user or authentication status from context

    // If the user is not authenticated, redirect to the login page
    return user ? children : <Navigate to="/Register" />;
};

export const OnlyGuestRoute = ({ children }) => {
    const { user } = useAuth();

    return user ? <Navigate to={'/dashboard'} /> : children
}
