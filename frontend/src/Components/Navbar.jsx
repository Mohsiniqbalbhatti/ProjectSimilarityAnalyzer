import React, { useEffect, useState } from "react";
import { FaHome, FaUserAlt } from "react-icons/fa";
import { Link, NavLink, useNavigate, useParams } from "react-router-dom"; // Import Link from react-router-dom
import axios from "axios";
import toast from "react-hot-toast";
import Loader from "./Loader";
import { useUser } from "../context/AuthUser";
import { useHistory } from "../context/getHistory";
function Navbar() {
  const [load, setLoad] = useState(false);
  const { user, setUser } = useUser();
  const navigate = useNavigate();
  const { history } = useHistory(); // Ensure you're getting history from the context
  const [screen, setScreen] = useState(); // State for screen size

  const handleLogout = async () => {
    setLoad(true);
    try {
      const res = await axios.get(`${import.meta.env.VITE_URL}logout`, {
        withCredentials: true,
      });
      if (res.data) {
        toast.success(res.data.message);
        setUser(null);
        navigate("/");
      }
    } catch (error) {
      console.log("error", error);
    } finally {
      setLoad(false);
    }
  };

  // Handle history item click and navigate to the details page
  const { id } = useParams();
  const handleHistoryClick = (id) => {
    navigate(`history/${id}`);
  };

  useEffect(() => {
    const updateScreen = () => {
      if (window.innerWidth > 992) {
        setScreen("lg");
      } else {
        setScreen("sm");
      }
    };

    updateScreen(); // Set initial screen size
    window.addEventListener("resize", updateScreen); // Add event listener for resizing

    // Cleanup event listener when component unmounts
    return () => window.removeEventListener("resize", updateScreen);
  }, []);

  return (
    <nav className="navbar bg-body-tertiary fixed-top ">
      {load && <Loader />}
      <div className="w-100">
        <div className="row d-flex">
          <div className="col-12 px-5 d-flex justify-content-cente align-items-center">
            <button
              className="navbar-toggler d-block d-lg-none"
              type="button"
              data-bs-toggle="offcanvas"
              data-bs-target="#offcanvasNavbar"
              aria-controls="offcanvasNavbar"
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
            <Link className="navbar-brand text-gradient ps-2 ps-lg-0" to="/">
              Project Similarity Analyzer
            </Link>

            {screen === "lg" ? (
              <ul className="d-flex justify-content-center align-items-center list-none my-auto ms-auto">
                <li className="nav-item py-2 py-lg-0">
                  <NavLink
                    className={({ isActive }) =>
                      isActive ? "link-nav nav-active" : "link-nav"
                    }
                    to="/"
                  >
                    Home
                  </NavLink>
                </li>

                <li>
                  <Link
                    className="user mx-2 btn btn-second ms-auto"
                    to={"/profile"}
                  >
                    <FaUserAlt /> {user?.username}
                  </Link>
                </li>
              </ul>
            ) : (
              ""
            )}
          </div>
        </div>

        {/*offcanvas side bar  */}
        <div
          className="offcanvas offcanvas-start"
          // style={{
          //   maxWidth: "300px",
          // }}
          tabIndex="-1"
          id="offcanvasNavbar"
          aria-labelledby="offcanvasNavbarLabel"
        >
          <div className="offcanvas-header d-flex flex-column ">
            <Link className="navbar-brand text-gradient ps-2 ps-lg-0" to="/">
              Project Similarity Analyzer
            </Link>
            <ul className="d-flex justify-content-center align-items-between list-none my-auto ">
              <li className="nav-item py-2 py-lg-0">
                <NavLink
                  className={({ isActive }) =>
                    isActive ? "link-nav nav-active" : "link-nav"
                  }
                  to="/"
                >
                  <FaHome />
                </NavLink>
              </li>

              <li>
                <Link
                  className="user mx-2 btn btn-second ms-auto"
                  to={"/profile"}
                >
                  <FaUserAlt />
                </Link>
              </li>
            </ul>
            <button
              type="button"
              className="btn-close"
              data-bs-dismiss="offcanvas"
              aria-label="Close"
            ></button>
          </div>
          <div className="offcanvas-body">
            <div className="menu">
              <ul className="px-0">
                {history?.map((hist) => (
                  <li
                    key={hist.id} // Added a key prop for uniqueness
                    className={`${
                      id == parseFloat(hist.id) ? "bg-800" : ""
                    } history-li mx-0`}
                    onClick={() => handleHistoryClick(hist.id)}
                  >
                    <span className="history-text">
                      <strong>Title: </strong> {hist?.title}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            <div style={{ backgroundColor: "rgba(255, 255, 255)" }}>
              <button
                className="btn btn-main w-75 position-absolute"
                style={{ bottom: 10 }}
                onClick={handleLogout}
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
