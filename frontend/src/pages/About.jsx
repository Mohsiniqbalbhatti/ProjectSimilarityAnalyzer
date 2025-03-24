import React from "react";
import { Link } from "react-router-dom";
import {
  FaRegLightbulb,
  FaCogs,
  FaBullseye,
  FaLaptopCode,
} from "react-icons/fa";

function About() {
  return (
    <div className="container py-5">
      <div className="row">
        <div className="col-md-12 text-center mb-5 ">
          <h2 className="text-gradient">About Project Similarity Analyzer</h2>
        </div>
        <div className="col-12">
          <p className="text-justify">
            Welcome to <strong>Project Similarity Analyzer</strong>, where
            innovation meets academic integrity. Established with the goal of
            transforming the research and academic landscape,{" "}
            <strong>Project Similarity Analyzer</strong> brings a fresh
            perspective to academic tools by blending cutting-edge AI technology
            with traditional academic values to deliver unparalleled support to
            students, researchers, and institutions.
          </p>

          <p className="text-justify">
            Founded by a group of passionate final-year students at{" "}
            <strong>GC University Faisalabad (GCUF)</strong>, we embarked on
            this journey with the mission to simplify and enhance the process of
            ensuring originality in academic work. Since our inception, we've
            been dedicated to helping students and researchers maintain the
            highest standards of academic honesty and originality.
          </p>

          <p className="text-justify">
            At <strong>Project Similarity Analyzer</strong>, we recognize that
            every research project is unique, and that's why we tailor our
            analysis to meet the distinct needs of each user. Whether you're a
            student working on your final year project, a researcher refining
            your academic paper, or an institution ensuring the integrity of its
            research, our advanced AI-powered tool is here to help. We make
            similarity checking seamless and efficient, ensuring your academic
            journey is as transparent and authentic as possible.
          </p>
        </div>
      </div>
    </div>
  );
}

export default About;
