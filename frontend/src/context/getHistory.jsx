import axios from "axios";
import { createContext, useContext, useEffect, useState } from "react";

const HistoryContext = createContext();

const getHistory = async (setHistory, setLoad) => {
  console.log("fetching history");
  setLoad(true);
  try {
    const res = await axios.get(`${import.meta.env.VITE_URL}history`, {
      withCredentials: true,
    });
    if (res.status === 200) {
      setHistory(res.data?.history);
    }
  } catch (error) {
    console.log("Error fetching history:", error);
  } finally {
    setLoad(false);
  }
};
export default getHistory;

export const HistoryProvider = ({ children }) => {
  const [load, setLoad] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    getHistory(setHistory, setLoad);
  }, []);
  return (
    <HistoryContext.Provider value={{ history, load, setHistory }}>
      {children}{" "}
    </HistoryContext.Provider>
  );
};

export const useHistory = () => useContext(HistoryContext);
