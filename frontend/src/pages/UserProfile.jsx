import React, { useState } from "react";
import { MdAlternateEmail } from "react-icons/md";
import { FaEye, FaKey, FaRegEyeSlash } from "react-icons/fa";
import axios from "axios";
import toast from "react-hot-toast";
import { useForm } from "react-hook-form";
import Loader from "../Components/Loader";
import { useUser } from "../context/AuthUser";

function UserProfile() {
  const { user } = useUser();
  const [load, setLoad] = useState(false); // State to handle loading
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showCurrentPassword1, setShowCurrentPassword1] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();
  const {
    register: registerPasswordChange,
    handleSubmit: handlePasswordChange,
    formState: { errors: errorsPasswordChange },
    getValues,
    reset,
  } = useForm();

  const OnEmailChange = async (data) => {
    const EmailData = {
      username: user.username,
      email: user.email,
      newEmail: data.newEmail,
      password: data.password,
    };
    console.log(EmailData);
    setLoad(true);
    try {
      // Await the axios POST request to ensure it completes
      const res = await axios.post(
        `${import.meta.env.VITE_URL}user/updateEmail`,
        EmailData,
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        }
      );

      if (res.data) {
        toast.success(res.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.response?.data?.error || "An error occurred");
    } finally {
      setLoad(false);
    }
  };
  const toggleCurrentPasswordVisibility = () => {
    setShowCurrentPassword(!showCurrentPassword);
  };
  const toggleCurrentPasswordVisibility1 = () => {
    setShowCurrentPassword1(!showCurrentPassword1);
  };
  const toggleNewPasswordVisibility = () => {
    setShowNewPassword(!showNewPassword);
  };
  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };
  const eyeplace = {
    position: "absolute",
    right: "30px",
    top: "67%", // Aligns to the center of the input field
    transform: "translateY(-50%)", // Centers vertically
    cursor: "pointer",
    zIndex: 2,
  };
  const invalidFeedback = {
    position: "absolute",
    bottom: "-20px",
    fontSize: "0.875em",
    color: "#dc3545",
    width: "100%",
    paddingLeft: "15px",
  };

  // change password
  const onSubmitPasswordChange = async (data) => {
    const passwordData = {
      userName: user.username,
      password: data.password,
      newPassword: data.newPassword,
    };
    setLoad(true);
    try {
      // Await the axios POST request to ensure it completes
      const res = await axios.post(
        `${import.meta.env.VITE_URL}user/updatePassword`,
        passwordData,
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        }
      );

      if (res.status === 200) {
        toast.success(res.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.response?.data?.error || "An error occurred");
    } finally {
      setLoad(false);
      reset();
    }
  };
  return (
    <div className="px-4 px-lg-2">
      {load && <Loader />}

      <div className="row p-2 my-2">
        <div className="col-12 my-2 bg-200 py-4 rounded">
          <h4>
            {" "}
            <MdAlternateEmail className="me-1" />
            Change Your email address
          </h4>
          <form onSubmit={handleSubmit(OnEmailChange)}>
            <div className="mb-3">
              <label>Current Email</label>
              <input
                type="text"
                className="form-control p-2"
                disabled
                placeholder={user?.email}
              />
            </div>
            <div className="mb-3">
              <label>New Email</label>
              <input
                type="text"
                className={`form-control p-2 ${
                  errors.newEmail ? "is-invalid" : ""
                }`} // Apply 'is-invalid' if there's an error
                id="new_email"
                placeholder="Enter New email"
                {...register("newEmail", { required: true })}
              />
              {errors.newEmail && (
                <div className="invalid-feedback">New Email is Required.</div>
              )}
            </div>
            <div className="mb-3 position-relative">
              <label>Current Password</label>
              <input
                type={!showCurrentPassword ? "password" : "text"}
                className={`form-control p-2 ${
                  errors.password ? "is-invalid" : ""
                }`} // Apply 'is-invalid' if there's an error
                id="current_password"
                placeholder="Enter Current Password"
                {...register("password", { required: true })}
              />
              {errors.password && (
                <div className="invalid-feedback">Password is Required.</div>
              )}
              <span style={eyeplace} onClick={toggleCurrentPasswordVisibility}>
                {!showCurrentPassword ? <FaRegEyeSlash /> : <FaEye />}
              </span>
            </div>

            <button type="submit" className="btn btn-main py-2">
              Change Email
            </button>
          </form>
        </div>
        <div className="col-12 my-2 bg-200 py-4 rounded">
          <h4>
            {" "}
            <FaKey className="me-1" />
            Change Your Password
          </h4>
          <form onSubmit={handlePasswordChange(onSubmitPasswordChange)}>
            <div className="mb-3 position-relative">
              <label>Current Password</label>
              <input
                type={!showCurrentPassword1 ? "password" : "text"}
                className={`form-control p-2 ${
                  errorsPasswordChange.password ? "is-invalid" : ""
                }`}
                placeholder="Enter Current Password"
                {...registerPasswordChange("password", {
                  required: "Current Password is required",
                })}
              />
              {errorsPasswordChange.password && (
                <div className="invalid-feedback" style={invalidFeedback}>
                  {errorsPasswordChange.password.message}
                </div>
              )}
              <span style={eyeplace} onClick={toggleCurrentPasswordVisibility1}>
                {!showCurrentPassword1 ? <FaRegEyeSlash /> : <FaEye />}
              </span>
            </div>

            <div className="mb-3 position-relative">
              <label>New Password</label>
              <input
                type={!showNewPassword ? "password" : "text"}
                className={`form-control p-2 ${
                  errorsPasswordChange.newPassword ? "is-invalid" : ""
                }`}
                placeholder="Enter New Password"
                {...registerPasswordChange("newPassword", {
                  required: "New Password is required",
                })}
              />
              {errorsPasswordChange.newPassword && (
                <div className="invalid-feedback" style={invalidFeedback}>
                  {errorsPasswordChange.newPassword.message}
                </div>
              )}
              <span style={eyeplace} onClick={toggleNewPasswordVisibility}>
                {!showNewPassword ? <FaRegEyeSlash /> : <FaEye />}
              </span>
            </div>

            <div className="mb-3 position-relative">
              <label>Confirm New Password</label>
              <input
                type={!showConfirmPassword ? "password" : "text"}
                className={`form-control p-2 ${
                  errorsPasswordChange.confirmPassword ? "is-invalid" : ""
                }`}
                placeholder="Confirm New Password"
                {...registerPasswordChange("confirmPassword", {
                  required: "Confirmation is required",
                  validate: (value) => {
                    const { newPassword } = getValues();
                    return (
                      value === newPassword || "New Passwords do not match"
                    );
                  },
                })}
              />
              {errorsPasswordChange.confirmPassword && (
                <div className="invalid-feedback" style={invalidFeedback}>
                  {errorsPasswordChange.confirmPassword.message}
                </div>
              )}
              <span style={eyeplace} onClick={toggleConfirmPasswordVisibility}>
                {!showConfirmPassword ? <FaRegEyeSlash /> : <FaEye />}
              </span>
            </div>

            <button type="submit" className="btn btn-warning py-2">
              Change Password
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default UserProfile;
