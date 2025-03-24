import React, { useState, useRef, useEffect } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";
import toast from "react-hot-toast";
import Loader from "../Components/Loader";
import { useUser } from "../context/AuthUser";
import { useNavigate, useParams } from "react-router-dom";

function History() {
  const { user } = useUser();
  const navigate = useNavigate();

  // Redirect if user is not logged in
  if (!user) {
    navigate("/");
    return null; // Prevent further rendering
  }

  const [load, setLoad] = useState(false); // Loading state for form submission
  const [matches, setMatches] = useState([]);
  const [noSimilar, setNoSimilar] = useState(false);
  const [checking, setChecking] = useState(false);
  const [history, setHistory] = useState(null);

  const { id } = useParams();
  const resultSectionRef = useRef(null);

  // Using React Hook Form for form validation
  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm();
  // Using React Hook Form for the feedback form
  const {
    register: registerFeedback,
    handleSubmit: handleSubmitFeedback,
    formState: { errors: feedbackErrors },
    reset: resetFeedback,
  } = useForm();
  // Handle form submission
  const onSubmit = async (data) => {
    setLoad(true);
    const projectData = {
      title: data.title,
      domain: data.domain,
      summary: data.summary,
      historyId: id,
    };

    try {
      const res = await axios.post(
        `${import.meta.env.VITE_URL}index`,
        projectData,
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        }
      );

      if (res.status === 200) {
        if (res?.data?.matches.length > 0) {
          setMatches(res.data.matches || []); // Set matches from response
          setNoSimilar(true);
          setChecking(true);
          toast.success(res.data.message || "Success!");
          // Scroll to result section after submission
          if (resultSectionRef.current) {
            resultSectionRef.current.scrollIntoView({ behavior: "smooth" });
          }
        } else {
          setNoSimilar(false); // No matches found
          setChecking(true);
        }
      } else {
        toast.error(res.data.error || "An error occurred.");
      }
    } catch (error) {
      console.error("Error", error);
      toast.error(error?.response?.data?.error || "Something went wrong!");
    } finally {
      setLoad(false); // End loading state
    }
  };

  const handleReset = () => {
    setChecking(true);
    reset(); // Resets the form to default values
  };

  const getHistory = async () => {
    setLoad(true);
    try {
      const res = await axios.get(`${import.meta.env.VITE_URL}/history/${id}`, {
        withCredentials: true,
      });
      if (res.status === 200) {
        setHistory(res.data);
        console.log(res.data);
        if (res.data.results.length > 0) {
          setMatches(res.data.results || []);
          setNoSimilar(true);
          setChecking(true);
        } else {
          setNoSimilar(false);
          setChecking(true);
        }

        // Reset form with fetched history data
        reset({
          title: res.data.title,
          domain: res.data.domain,
          summary: res.data.summary,
        });
      }
    } catch (error) {
      console.log("error", error);
    } finally {
      setLoad(false);
    }
  };

  useEffect(() => {
    setChecking(false);
    setNoSimilar(false);
    setHistory(null);
    setMatches([]);
    getHistory();
  }, [id]);

  // Handle feedback form submission
  const onFeedbackSubmit = async (data) => {
    const feedbackData = {
      message: data.feedback,
      historyId: history.id,
    };
    setLoad(true);
    try {
      const res = await axios.post(
        `${import.meta.env.VITE_URL}feedback`,
        feedbackData,
        {
          headers: { "Content-Type": "application/json" },
          withCredentials: true,
        }
      );
      if (res.status === 200) {
        toast.success(res?.data?.message || "Thankyou for your Feedback!");
        resetFeedback(); // Reset feedback form after submission
      }
    } catch {
      console.log("error", errors);
    } finally {
      setLoad(false);
    }
  };

  return (
    <>
      <div className="row mt-10">
        {load && <Loader />}
        <div className="col-12">
          <h3 className="text-center">
            Check the Similarity of your Project Now!
          </h3>
        </div>
        <div className="col-12 mb-5">
          <form
            className="card p-2 p-md-5 glass d-flex flex-column justify-content-center"
            onSubmit={handleSubmit(onSubmit)}
          >
            {/* Project Title */}
            <div className="mb-3">
              <label htmlFor="title" className="form-label">
                Title of your Project
              </label>
              <input
                type="text"
                className={`form-control ${errors.title ? "is-invalid" : ""}`}
                id="title"
                placeholder="Enter Project title here"
                {...register("title", {
                  required: "Title is required",
                  minLength: {
                    value: 5,
                    message: "Title must be at least 5 characters",
                  },
                })}
              />
              {errors.title && (
                <div className="invalid-feedback">{errors.title.message}</div>
              )}
            </div>

            {/* Domain */}
            <div className="mb-3">
              <label htmlFor="domain" className="form-label">
                Domain
              </label>
              <input
                type="text"
                className={`form-control ${errors.domain ? "is-invalid" : ""}`}
                id="domain"
                placeholder="Such as communication"
                {...register("domain", { required: "Domain is required" })}
              />
              {errors.domain && (
                <div className="invalid-feedback">{errors.domain.message}</div>
              )}
            </div>

            {/* Project Summary */}
            <div className="mb-3">
              <label htmlFor="summary" className="form-label">
                Concise Summary of your Project
              </label>
              <textarea
                className={`form-control ${errors.summary ? "is-invalid" : ""}`}
                id="summary"
                rows="3"
                {...register("summary", {
                  required: "Summary is required",
                  minLength: {
                    value: 30,
                    message: "Summary must be at least 30 characters",
                  },
                })}
              />
              {errors.summary && (
                <div className="invalid-feedback">{errors.summary.message}</div>
              )}
            </div>

            {/* Submit Button */}
            <div className=" ">
              <button
                type="submit"
                className="btn btn-main me-2"
                disabled={load}
              >
                Check Similarity
              </button>
              <button
                type="reset"
                className="btn btn-secondary"
                disabled={load}
                onClick={handleReset}
              >
                Reset
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Scroll to Result Section */}
      <div className="row" ref={resultSectionRef}>
        <div className="col-12">
          {checking && noSimilar ? (
            <div className="card mt-2  glass">
              <div className="d-flex justify-content-between flex-column flex-md-row ">
                <h4 className="">Results for your Project Similarity Check</h4>
                <button
                  className="btn btn-main  mt-3 mt-md-0"
                  type="button"
                  data-bs-toggle="modal"
                  data-bs-target="#feedback"
                  style={{ maxWidth: "180px" }}
                >
                  FeedBack
                </button>
              </div>
              <ul className="mt-3 mt-md-0">
                <li>
                  Found Similarity with <strong>{matches.length}</strong> number
                  of Projects.
                </li>
              </ul>
              <div className="accordion" id="accordionExample">
                {matches.map((match, index) => (
                  <div className="accordion-item" key={index}>
                    <h2 className="accordion-header">
                      <button
                        className="accordion-button"
                        type="button"
                        data-bs-toggle="collapse"
                        data-bs-target={`#collapse${index}`}
                        aria-expanded={index === 0 ? "true" : "false"}
                        aria-controls={`collapse${index}`}
                      >
                        {/* Flex layout for title and similarity */}
                        <div className="d-flex flex-column">
                          <span className="py-2">
                            <strong>Title: </strong>
                            {match.title}
                          </span>
                          <span className="py-2">
                            {" "}
                            <strong>Similarity: </strong>
                            {`${match.similarity}%`}
                          </span>
                          <span className="py-2">
                            {" "}
                            <strong>Domain: </strong>
                            {`${match.domain}`}
                          </span>
                        </div>
                      </button>
                    </h2>
                    <div
                      id={`collapse${index}`}
                      className="accordion-collapse collapse"
                      data-bs-parent="#accordionExample"
                    >
                      <div className="accordion-body">{match.summary}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : checking && !noSimilar ? (
            <div className="card mt-1 mb-3 py-3">
              <h6 className="text-center m-3">
                "Well done! Your project doesn't match any existing ones, making
                it truly unique."
              </h6>
            </div>
          ) : (
            ""
          )}
        </div>
        {/* Feedback Modal */}

        <div
          className="modal fade"
          id="feedback"
          data-bs-backdrop="static"
          data-bs-keyboard="false"
          tabIndex="-1"
          aria-labelledby="staticBackdropLabel"
          aria-hidden="true"
        >
          <div className="modal-dialog modal-dialog-centered">
            <div className="modal-content">
              <div className="modal-header">
                <h1 className="modal-title fs-5" id="staticBackdropLabel">
                  Feedback
                </h1>
                <button
                  type="button"
                  className="btn-close"
                  data-bs-dismiss="modal"
                  aria-label="Close"
                ></button>
              </div>
              <div className="modal-body">
                <form onSubmit={handleSubmitFeedback(onFeedbackSubmit)}>
                  <div className="mb-3">
                    <label htmlFor="feedback" className="form-label">
                      Your Honest Feedback
                    </label>
                    <textarea
                      className={`form-control ${
                        feedbackErrors.feedback ? "is-invalid" : ""
                      }`}
                      id="feedback"
                      rows="3"
                      placeholder="Feedback"
                      {...registerFeedback("feedback", {
                        required: "Feedback is required",
                        minLength: {
                          value: 10,
                          message: "Feedback must be at least 30 characters",
                        },
                      })}
                    />
                    {feedbackErrors.feedback && (
                      <div className="invalid-feedback">
                        {feedbackErrors.feedback.message}
                      </div>
                    )}
                  </div>
                  <div className="modal-footer">
                    <button
                      type="button"
                      className="btn btn-secondary"
                      data-bs-dismiss="modal"
                    >
                      Close
                    </button>
                    <button type="submit" className="btn btn-main">
                      Submit Feedback
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default History;
