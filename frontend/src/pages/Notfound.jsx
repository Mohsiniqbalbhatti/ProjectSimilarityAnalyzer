import React from "react";
import { Link } from "react-router-dom";

function Notfound() {
  return (
    <div className="container">
      <div className="d-flex flex-column justify-content-center align-items-center vh-100">
        <h1 className="display-4">404 - Page Not Found</h1>
        <p className="lead">The page you are looking for does not exist.</p>
        <Link to="/" className="btn btn-main">
          Go back to Home
        </Link>
      </div>
    </div>
  );
}

export default Notfound;
