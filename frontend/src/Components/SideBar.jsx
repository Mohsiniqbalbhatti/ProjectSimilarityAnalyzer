import React, { useState } from "react";
import toast from "react-hot-toast";
import { useNavigate, useParams } from "react-router-dom";
import { useUser } from "../context/AuthUser";
import axios from "axios";
import Loader from "./Loader";
import { useHistory } from "../context/getHistory";

function SideBar() {
  const [load, setLoad] = useState(false);
  const { setUser } = useUser();
  const navigate = useNavigate();
  const { history } = useHistory(); // Ensure you're getting history from the context

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
  const handleHistoryClick = (id) => {
    navigate(`history/${id}`);
  };

  const { id } = useParams();

  return (
    <div>
      {load && <Loader />}
      {/* Sidebar */}
      <div
        className="sideBar py-3 position-fixed mt-5"
        style={{ height: "100vh", minWidth: "22%", zIndex: "1000" }}
      >
        {/* Middle Section: Scrollable Menu */}
        <div
          className="col-12 menu"
          style={{
            overflowY: "auto",
            maxHeight: "calc(100vh - 100px)",
            paddingBottom: "60px",
          }}
        >
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

        {/* Bottom Section: Logout Button */}
        <div
          className="position-absolute py-1 d-flex justify-content-center"
          style={{
            bottom: 50,
            backgroundColor: "rgba(255, 255, 255)",
            zIndex: 10,
            width: "100%",
          }}
        >
          <button className="btn btn-main w-100 mx-3" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </div>
    </div>
  );
}

export default SideBar;
