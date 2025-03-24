import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  FaCheckCircle,
  FaCogs,
  FaEye,
  FaFileUpload,
  FaRegEyeSlash,
} from "react-icons/fa";
import { useForm } from "react-hook-form";
import Loader from "../Components/Loader";
import axios from "axios";
import faq from "../assets/faq.png";
import toast from "react-hot-toast";
import { useUser } from "../context/AuthUser";
import { useHistory } from "../context/getHistory";
import p1 from "../assets/p1.png";
import p2 from "../assets/p2.png";
import p3 from "../assets/p3.png";
import Testimonial from "../Components/Testimonial";

function Login() {
  const [load, setLoad] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const navigate = useNavigate();
  const { setUser } = useUser();
  const { setHistory } = useHistory();
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm();
  const onSubmit = async (data) => {
    const userinfo = {
      identifier: data.email,
      password: data.password,
    };

    setLoad(true); // Start loading state

    try {
      const res = await axios.post(
        `${import.meta.env.VITE_URL}login`,
        userinfo,
        {
          withCredentials: true,
          headers: { "Content-Type": "application/json" },
        }
      );

      if (res.status === 200) {
        if (res.data) {
          console.log(res?.data);
          toast.success(res?.data?.message);

          // 🔧 Delay to let cookie store
          await new Promise((resolve) => setTimeout(resolve, 500));

          const getuser = await axios.get(`${import.meta.env.VITE_URL}user`, {
            withCredentials: true,
          });

          if (getuser.data.user) {
            setUser(getuser?.data?.user);

            const getHistory = await axios.get(
              `${import.meta.env.VITE_URL}history`,
              {
                withCredentials: true,
              }
            );
            if (getHistory.data.history) {
              setHistory(getHistory?.data?.history);
            }
          }
        }
      } else {
        toast.error(res.data.error || "Invalid email or password");
      }
    } catch (error) {
      console.error("Login Error", error);
      toast.error(error?.response?.data?.error || "Something went wrong!");
    } finally {
      setLoad(false); // End loading state
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };
  const password = watch("password");

  useEffect(() => {
    if (password && password.length > 0) {
      setIsTyping(true);
    } else {
      setIsTyping(false);
    }
  }, [password]);

  const eyeplace = {
    position: "absolute",
    right: "20px",
    top: "50%",
    transform: "translateY(-50%)",
    cursor: "pointer",
  };

  return (
    <>
      {load && <Loader />}
      <div className="row justify-content-center ">
        <div className="col-12 mt-5 pt-5">
          <h2 className=" text-center my-2  py-1 mt-5 pt-5 ">
            Check Project Similarity Instantly
          </h2>
        </div>
        <div className="col-12 mt-0 ">
          <h3 className="text-gradient text-center my-2">
            Dominate Your Research with the World’s Best Project Similarity
            Analyzer at Unbeatable Accuracy!
          </h3>
        </div>
        <div className="col-12 d-flex justify-content-center my-3">
          <button className="btn btn-main me-2">Contact Us</button>
          <button className="btn btn-second ms-2">Signup</button>
        </div>
      </div>
      <div className="row justify-content-center mt-4">
        <div className="col-11  bg-300 py-5 border-400  px-2 px-sm-5 rounded">
          <form onSubmit={handleSubmit(onSubmit)}>
            <div className="row g-3 justify-content-center ">
              <div className="col-lg-3  ">
                <input
                  type="email"
                  className="form-control p-3"
                  id="exampleInputEmail1"
                  placeholder="Email"
                  {...register("email", { required: true })}
                />
                {errors.email && (
                  <div className="invalid-feedback">Email is required.</div>
                )}{" "}
              </div>
              <div className="col-lg-3 position-relative">
                <input
                  type={!showPassword ? "password" : "text"}
                  className="form-control p-3"
                  id="exampleInputPassword1"
                  placeholder="Password"
                  {...register("password", { required: true })}
                />
                {isTyping && (
                  <span style={eyeplace} onClick={togglePasswordVisibility}>
                    {!showPassword ? <FaRegEyeSlash /> : <FaEye />}
                  </span>
                )}
                {errors.password && (
                  <div className="invalid-feedback">Password is required.</div>
                )}{" "}
              </div>

              <div className="col-lg-2 d-flex align-items-center">
                <button type="submit" className="btn btn-main w-100  p-3">
                  Login
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      <div className="row justify-content-center mt-4">
        <div className="col-12 mt-5">
          <h3 className="text-center mt-3">
            Why Choose Project Similarity Analyzer?
          </h3>
        </div>
        <div className="col-12 px-2 d-flex justify-content-evenly align-items-center my-3 flex-column flex-lg-row ">
          <div
            className="card my-3 my-lg-0"
            style={{ width: "19rem", minHeight: "235px" }}
          >
            <div className="card-body text-center">
              <FaFileUpload size={40} color="#B61B00" />
              <h5 className="card-title mt-2">Submit Your Project</h5>
              <p className="card-text">
                Enter your project title and summary on our platform.
              </p>
              <a className="btn btn-main">Start Now</a>
            </div>
          </div>

          <div
            className="card my-3 my-lg-0"
            style={{ width: "19rem", minHeight: "235px" }}
          >
            <div className="card-body text-center">
              <FaCogs size={40} color="#B61B00" />
              <h5 className="card-title mt-2">AI Comparison</h5>
              <p className="card-text">
                Our AI system scans your project and compares it with a vast
                database of previous projects.
              </p>
              <a className="btn btn-main">See Results</a>
            </div>
          </div>

          <div
            className="card my-3 my-lg-0"
            style={{ width: "19rem", minHeight: "235px" }}
          >
            <div className="card-body text-center">
              <FaCheckCircle size={40} color="#B61B00" />
              <h5 className="card-title mt-2">Get Your Similarity Score</h5>
              <p className="card-text">
                Receive an instant similarity score that shows how closely your
                project matches others.
              </p>
              <a className="btn btn-main">Refine Your Project</a>
            </div>
          </div>
        </div>
      </div>
      <div className="row justify-content-center mt-4">
        <div className="col-11 bg-300 py-5 border-400 px-2 px-sm-5 rounded">
          <Testimonial />
        </div>
      </div>
      <section id="faq">
        <div className="row my-5 justify-content-center align-items-center">
          <div className="col-12">
            <h1 className="text-center">
              Frequently Asked Question <span className="text-red">(FAQ)</span>
            </h1>
          </div>
          <div className="col-12 col-md-10 col-lg-5 my-3 mx-auto">
            <div
              className="accordion accordion-flush"
              id="accordionFlushExample"
            >
              {/* FAQ 1 */}
              <div
                className="accordion-item"
                style={{
                  margin: "0 15px 20px 0",
                  borderRadius: "10px",
                  border: "1px solid #ffffff20",
                  boxShadow:
                    "0 14px 28px rgba(0, 0, 0, .25), 0 10px 10px rgba(0, 0, 0, .22)",
                  overflow: "hidden",
                }}
              >
                <h2 className="accordion-header">
                  <button
                    className="accordion-button collapsed"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#flush-collapseOne"
                    aria-expanded="false"
                    aria-controls="flush-collapseOne"
                    style={{
                      border: "none",
                      backgroundColor: "transparent",
                      outline: "none",
                    }}
                  >
                    What is the Project Similarity Analyzer?
                  </button>
                </h2>
                <div
                  id="flush-collapseOne"
                  className="accordion-collapse collapse"
                  data-bs-parent="#accordionFlushExample"
                >
                  <div className="accordion-body">
                    Project Similarity Analyzer is a tool that helps ensure the
                    originality of your project ideas by comparing titles and
                    summaries with existing work in the database.
                  </div>
                </div>
              </div>

              {/* FAQ 2 */}
              <div
                className="accordion-item"
                style={{
                  margin: "0 15px 20px 0",
                  borderRadius: "10px",
                  border: "1px solid #ffffff20",
                  boxShadow:
                    "0 14px 28px rgba(0, 0, 0, .25), 0 10px 10px rgba(0, 0, 0, .22)",
                  overflow: "hidden",
                }}
              >
                <h2 className="accordion-header">
                  <button
                    className="accordion-button collapsed"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#flush-collapseTwo"
                    aria-expanded="false"
                    aria-controls="flush-collapseTwo"
                    style={{
                      border: "none",
                      backgroundColor: "transparent",
                      outline: "none",
                    }}
                  >
                    How does the analyzer calculate similarity?
                  </button>
                </h2>
                <div
                  id="flush-collapseTwo"
                  className="accordion-collapse collapse"
                  data-bs-parent="#accordionFlushExample"
                >
                  <div className="accordion-body">
                    The analyzer uses advanced AI and Natural Language
                    Processing (NLP) techniques to compare project titles and
                    summaries, providing a percentage-based similarity score.
                  </div>
                </div>
              </div>

              {/* FAQ 3 */}
              <div
                className="accordion-item"
                style={{
                  margin: "0 15px 20px 0",
                  borderRadius: "10px",
                  border: "1px solid #ffffff20",
                  boxShadow:
                    "0 14px 28px rgba(0, 0, 0, .25), 0 10px 10px rgba(0, 0, 0, .22)",
                  overflow: "hidden",
                }}
              >
                <h2 className="accordion-header">
                  <button
                    className="accordion-button collapsed"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#flush-collapseThree"
                    aria-expanded="false"
                    aria-controls="flush-collapseThree"
                    style={{
                      border: "none",
                      backgroundColor: "transparent",
                      outline: "none",
                    }}
                  >
                    Who can use the Project Similarity Analyzer?
                  </button>
                </h2>
                <div
                  id="flush-collapseThree"
                  className="accordion-collapse collapse"
                  data-bs-parent="#accordionFlushExample"
                >
                  <div className="accordion-body">
                    The tool is designed for students, researchers, and academic
                    institutions who need to verify the originality of project
                    proposals or research work.
                  </div>
                </div>
              </div>

              {/* FAQ 4 */}
              <div
                className="accordion-item"
                style={{
                  margin: "0 15px 20px 0",
                  borderRadius: "10px",
                  border: "1px solid #ffffff20",
                  boxShadow:
                    "0 14px 28px rgba(0, 0, 0, .25), 0 10px 10px rgba(0, 0, 0, .22)",
                  overflow: "hidden",
                }}
              >
                <h2 className="accordion-header">
                  <button
                    className="accordion-button collapsed"
                    type="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#flush-collapseFour"
                    aria-expanded="false"
                    aria-controls="flush-collapseFour"
                    style={{
                      border: "none",
                      backgroundColor: "transparent",
                      outline: "none",
                    }}
                  >
                    Is my data stored or shared?
                  </button>
                </h2>
                <div
                  id="flush-collapseFour"
                  className="accordion-collapse collapse"
                  data-bs-parent="#accordionFlushExample"
                >
                  <div className="accordion-body">
                    No, your data is not stored or shared. The system only uses
                    your project details for comparison and does not retain any
                    sensitive information.
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="col-5 d-none d-lg-flex">
            <img src={faq} className="img-fluid" alt="FAQ" />
          </div>
        </div>
      </section>
      <div className="row">
        <div className="col-12 bg-200 bg-soft py-3">
          <p className="text-center">
            All Rights Reserved by &copy; Project Similarity Analyzer
          </p>
        </div>
      </div>
    </>
  );
}
export default Login;
