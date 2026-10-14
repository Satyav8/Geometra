import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The dev server forwards the API calls to the backend, so the browser only ever
    // talks to localhost:5173 and the request is same-origin. Without this the browser
    // would call the backend directly, need a CORS entry for localhost, and the backend
    // refuses exactly that entry when APP_ENV=production - leaving no way to point a
    // local UI at a production-configured backend. Change the target, not the frontend,
    // if the backend is on another port.
    proxy: {
      "/chat": "http://localhost:8000",
      "/session": "http://localhost:8000",
      "/unknown-questions": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
