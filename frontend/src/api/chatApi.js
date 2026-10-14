import axios from "axios";

// Same origin by default. Both ways of running the app now serve the UI and the API
// under ONE origin - the container's nginx proxies /chat to the backend, and `npm run
// dev` proxies it through Vite (see vite.config.js) - so the browser never makes a
// cross-origin request and CORS never has to be configured to get the chat working.
//
// That matters beyond convenience: under APP_ENV=production the backend REFUSES to start
// if CORS_ORIGINS contains localhost, so "just add localhost to CORS" was not available
// as a local workaround. Same-origin sidesteps the whole question.
//
// ?? and not ||: an explicitly empty VITE_API_URL means "same origin" and must be kept,
// where || would discard it and fall back to the default.
//
// Set VITE_API_URL only when the UI is genuinely served from a different origin than the
// API, which is how the Render deployment works (static site + separate web service).
const API_BASE_URL = import.meta.env.VITE_API_URL ?? "";

export const sendMessage = async (session_id, query, turn_number) => {
  const response = await axios.post(`${API_BASE_URL}/chat`, {
    session_id,
    query,
    turn_number,
  });
  return response.data; // ChatResponse object
};
