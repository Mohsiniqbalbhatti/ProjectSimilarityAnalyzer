import { createBrowserRouter } from "react-router-dom";
import App from "../App";
import Login from "../guest/Login";
import Guest from "../guest/Guest";
import Notfound from "../pages/Notfound";
import Dashboard from "../pages/Dashboard";
import History from "../pages/history";
import UserProfile from "../pages/UserProfile";
import Contact from "../pages/Contact";
import Signup from "../guest/Signup";
import About from "../pages/About";

const router = createBrowserRouter([
  {
    path: "",
    element: <App />,
    children: [
      { path: "/", element: <Dashboard /> },
      { path: "/profile", element: <UserProfile /> },
      { path: "/history/:id", element: <History /> },
    ],
  },
  {
    path: "/",
    element: <Guest />,
    children: [
      { path: "/", element: <Login /> },
      { path: "/contact", element: <Contact /> }, // Contact page also for guests
      { path: "/signup", element: <Signup /> }, // Contact page also for guests
      { path: "/about", element: <About /> }, // Contact page also for guests
    ],
  },
  { path: "*", element: <Notfound /> },
]);

export default router;
