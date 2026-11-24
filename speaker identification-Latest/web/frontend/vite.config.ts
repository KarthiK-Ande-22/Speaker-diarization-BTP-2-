import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development (npm run dev) the API calls go to the Flask server on port 5000.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:5000" } },
});
