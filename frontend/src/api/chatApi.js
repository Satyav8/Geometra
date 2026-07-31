import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const sendMessage = async (session_id, query, turn_number) => {
  const response = await axios.post(`${API_BASE_URL}/chat`, {
    session_id,
    query,
    turn_number,
  });
  return response.data; // ChatResponse object
};
