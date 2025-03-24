import React from "react";
import Slider from "react-slick";
// Add these to your main JS file (like index.js or App.js)
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

// Import your images
import p1 from "../assets/p1.png"; // Adjust the path as per your directory structure
import p2 from "../assets/p2.png";
import p3 from "../assets/p3.png";

function Testimonial() {
  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
  };

  return (
    <>
      <div className="container-fluid">
        <div className="slider-container">
          <Slider {...settings}>
            <div>
              <div className="carousel-item d-flex bg-200 flex-column active">
                <img src={p1} className="d-block pfp" alt="User 1" />
                <p className="text-justify">
                  “Project Similarity Analyzer helped me ensure that my research
                  project was completely original. I was able to detect any
                  similarities with other projects and refine my work
                  accordingly. It’s a game-changer for academic integrity!”
                </p>
                <strong>Ali Khan</strong>
                <p>Researcher at GCUF</p>
              </div>
            </div>
            <div>
              <div className="carousel-item d-flex bg-200 flex-column">
                <img src={p2} className="d-block pfp" alt="User 2" />
                <p className="text-justify">
                  “As a student at NUST, I used the Project Similarity Analyzer
                  to ensure my final year project was unique. The tool was
                  intuitive, and I was able to submit my work with confidence,
                  knowing it met all originality standards.”
                </p>
                <strong>Zafar Iqbal</strong>
                <p>Student at NUST</p>
              </div>
            </div>
            <div>
              <div className="carousel-item d-flex bg-200 flex-column">
                <img src={p3} className="d-block pfp" alt="User 3" />
                <p className="text-justify">
                  “The Project Similarity Analyzer gave me peace of mind while
                  working on my business project. It helped me identify
                  potential plagiarism risks and gave me insights into how to
                  improve my work. A must-have for any researcher.”
                </p>
                <strong>Ahmad Shah</strong>
                <p>Business Student at LUMS</p>
              </div>
            </div>
          </Slider>
        </div>
      </div>
    </>
  );
}

export default Testimonial;
