function Loader() {
  return (
    <div
      style={{
        position: "fixed", // Make the loader cover the full screen
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(255, 255, 255, 0.8)", // Optional: Add a semi-transparent background
        zIndex: 9999, // Ensure it sits above other elements
      }}
      className="d-flex justify-content-center align-items-center"
    >
      <span className="loader"></span>
    </div>
  );
}

export default Loader;
