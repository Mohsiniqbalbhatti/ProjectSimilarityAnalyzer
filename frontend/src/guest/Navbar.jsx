import React from "react";
import { Link, NavLink } from "react-router-dom";
function Navbar() {
  return (
    <header>
      <nav className="navbar navbar-expand-lg bg-2 nav-guest py-3 fixed-top">
        <div className="container-fluid">
          <Link className="navbar-brand text-gradient" to="/">
            Project Similarity Analyzer
          </Link>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
            aria-controls="navbarNav"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarNav">
            <ul className="navbar-nav ms-auto">
              <li className="nav-item py-2 py-lg-0">
                <NavLink
                  className={({ isActive }) =>
                    isActive ? "link-nav nav-active" : "link-nav"
                  }
                  to=""
                >
                  Login
                </NavLink>
              </li>
              {/* <li className="nav-item py-2 py-lg-0">
                <NavLink className="link-nav" to="blog">
                  Blog
                </NavLink>
              </li> */}
              <li className="nav-item py-2 py-lg-0">
                <NavLink className="link-nav" to="contact">
                  Contact
                </NavLink>
              </li>
              <li className="nav-item py-2 py-lg-0">
                <NavLink className="link-nav" to="about">
                  About
                </NavLink>
              </li>

              <li className="nav-item py-2 py-lg-0">
                <NavLink className="link-nav nav-active" to="signup">
                  Signup
                </NavLink>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </header>
  );
}

export default Navbar;
