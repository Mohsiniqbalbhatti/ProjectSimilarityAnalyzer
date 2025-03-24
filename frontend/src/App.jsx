import { Outlet } from "react-router-dom";
import Guest from "./guest/Guest";
import { Toaster } from "react-hot-toast";
import Navbar from "./Components/Navbar";
import SideBar from "./Components/SideBar";
import Loader from "./Components/Loader";
import { useUser } from "./context/AuthUser";

function App() {
  const { user, load } = useUser();
  if (load) {
    return <Loader />;
  }

  return (
    <div className="container-fluid">
      {user ? (
        <div className="row">
          <div className="col-12 ">
            <Navbar />
          </div>
          <div className="col-3 d-none d-lg-block">
            <SideBar />
          </div>
          <div className="col-12 col-lg-9 mt-10">
            {" "}
            <Outlet />
            <Toaster />
          </div>
        </div>
      ) : (
        <Guest />
      )}
    </div>
  );
}

export default App;
