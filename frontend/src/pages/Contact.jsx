import React, { useState } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";
import toast from "react-hot-toast";
import Loader from "../Components/Loader"; // Assuming you have a Loader component

function Contact() {
  const [load, setLoad] = useState(false); // Loading state
  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm();

  // Handle form submission
  const onSubmit = async (data) => {
    const contactData = {
      name: data.name,
      message: data.message,
    };

    setLoad(true); // Show loading spinner
    try {
      const res = await axios.post(
        `${import.meta.env.VITE_URL}contact`, // Backend endpoint
        contactData,
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        }
      );

      if (res.status === 200) {
        toast.success(res.data?.message);
        reset(); // Reset form after successful submission
      } else {
        toast.error(res.data.message || "Something went wrong!");
      }
    } catch (error) {
      toast.error(error.response?.data?.message || "An error occurred.");
    } finally {
      setLoad(false); // Hide loading spinner
    }
  };

  return (
    <>
      <div className="row justify-content-center">
        <div className="home-section-banner"></div>
        <div className="col-10 col-lg-8 my-2 bg-200 py-4 rounded">
          <h4>Contact Us</h4>
          {load && <Loader />} {/* Show loading spinner while submitting */}
          <form onSubmit={handleSubmit(onSubmit)}>
            {/* Name Field */}
            <div className="mb-3">
              <label>Name</label>
              <input
                type="text"
                className={`form-control p-2 ${
                  errors.name ? "is-invalid" : ""
                }`}
                id="name"
                placeholder="Enter your name"
                {...register("name", { required: "Name is required" })}
              />
              {errors.name && (
                <div className="invalid-feedback">{errors.name.message}</div>
              )}
            </div>

            {/* Message Field */}
            <div className="mb-3">
              <label>Message</label>
              <textarea
                className={`form-control p-2 ${
                  errors.message ? "is-invalid" : ""
                }`}
                id="message"
                placeholder="Enter your message"
                rows="4"
                {...register("message", { required: "Message is required" })}
              ></textarea>
              {errors.message && (
                <div className="invalid-feedback">{errors.message.message}</div>
              )}
            </div>

            {/* Submit Button */}
            <button type="submit" className="btn btn-main py-2">
              Send Message
            </button>
          </form>
        </div>
      </div>
    </>
  );
}

export default Contact;
