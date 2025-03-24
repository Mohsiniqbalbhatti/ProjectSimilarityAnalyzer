import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { RouterProvider } from "react-router-dom";
import router from "./Routes/router";
import { UserProvider } from "./context/AuthUser";
import { HistoryProvider } from "./context/getHistory";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <UserProvider>
      <HistoryProvider>
        <RouterProvider router={router} />
      </HistoryProvider>{" "}
    </UserProvider>
  </StrictMode>
);
