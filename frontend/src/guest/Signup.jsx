import axios from "axios";
import React, { useState } from "react";
import { useForm } from "react-hook-form";
import toast from "react-hot-toast";
import { FaEye, FaRegEyeSlash } from "react-icons/fa";
import { Link, useNavigate } from "react-router-dom";
import { FaArrowLeftLong } from "react-icons/fa6";
import Loader from "../Components/Loader";

function Signup() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [load, setLoad] = useState(false);
  const navigate = useNavigate();
  const {
    register,
    handleSubmit,
    getValues,
    formState: { errors },
  } = useForm();

  const onSubmit = async (data) => {
    setLoad(true); // Start loading state
    const userinfo = {
      name: data.name,
      username: data.username,
      email: data.email,
      password: data.password,
    };

    try {
      const res = await axios.post(
        `${import.meta.env.VITE_URL}signup`,
        userinfo
      );
      if (res.status === 200) {
        toast.success(res.data.message || "User registered successfully");
        navigate("/");
      } else {
        toast.error(res.data.error);
      }
    } catch (error) {
      console.log("error", error);
      toast.error(error.response.data.error);
    } finally {
      setLoad(false);
    }
  };
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };
  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };
  const eyeplace = {
    position: "absolute",
    right: "20px",
    top: "70%",
    transform: "translateY(-50%)",
    cursor: "pointer",
  };
  const invalidFeedback = {
    position: "absolute",
    bottom: "-20px",
    fontSize: "0.875em",
    color: "#dc3545",
    width: "100%",
    paddingLeft: "15px",
  };

  return (
    <div className="container">
      <div className="row justify-content-center">
        {load && <Loader />}
        <div className="home-section-banner"></div>

        <div className="col-11  col-md-8 col-lg-5 my-5 ">
          <Link className="mb-2 text-dark" to={"/"}>
            <FaArrowLeftLong /> Go back to Login page
          </Link>
          <h2 className="text-center my-4">Sign Up</h2>
          <form onSubmit={handleSubmit(onSubmit)}>
            {/* Full Name */}
            <div className="mb-3">
              <label htmlFor="name" className="form-label">
                Full Name
              </label>
              <input
                type="text"
                className="form-control p-2"
                id="name"
                {...register("name", { required: "Name is required" })}
              />
              {errors.fullname && (
                <span className="text-danger">{errors.name.message}</span>
              )}
            </div>
            {/* USer Name */}
            <div className="mb-3">
              <label htmlFor="username" className="form-label">
                Username
              </label>
              <input
                type="text"
                className="form-control p-2"
                id="username"
                {...register("username", { required: "Username is required" })}
              />
              {errors.username && (
                <span className="text-danger">{errors.username.message}</span>
              )}
            </div>

            {/* Email */}
            <div className="mb-3">
              <label htmlFor="email" className="form-label">
                Email
              </label>
              <input
                type="email"
                className="form-control p-2"
                id="email"
                {...register("email", {
                  required: "Email is required",
                  pattern: {
                    value: /^\S+@\S+$/i,
                    message: "Invalid email address",
                  },
                })}
              />
              {errors.email && (
                <span className="text-danger">{errors.email.message}</span>
              )}
            </div>

            {/* Password */}
            <div className="mb-3 position-relative">
              <label htmlFor="password" className="form-label">
                Password
              </label>
              <input
                type={!showPassword ? "password" : "text"}
                className="form-control p-2"
                id="password"
                {...register("password", { required: "Password is required" })}
              />
              <span style={eyeplace} onClick={togglePasswordVisibility}>
                {!showPassword ? <FaRegEyeSlash /> : <FaEye />}
              </span>
              {errors.password && (
                <span className="text-danger" style={invalidFeedback}>
                  {errors.password.message}
                </span>
              )}
            </div>

            {/* Confirm Password */}
            <div className="mb-3 position-relative">
              <label htmlFor="confirmPassword" className="form-label">
                Confirm Password
              </label>
              <input
                type={!showConfirmPassword ? "password" : "text"}
                className="form-control p-2"
                id="confirmPassword"
                {...register("confirmPassword", {
                  required: "Please confirm your password",
                  validate: (value) => {
                    const { password } = getValues();
                    return value === password || "Passwords do not match";
                  },
                })}
              />
              <span style={eyeplace} onClick={toggleConfirmPasswordVisibility}>
                {!showConfirmPassword ? <FaRegEyeSlash /> : <FaEye />}
              </span>
              {errors.confirmPassword && (
                <span className="text-danger" style={invalidFeedback}>
                  {errors.confirmPassword.message}
                </span>
              )}
            </div>

            <button type="submit" className="btn btn-main">
              Sign Up
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Signup;
