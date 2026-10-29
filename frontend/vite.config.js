// Vite config: dev server on 5173, API calls proxied to the FastAPI backend on 8000.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.BACKEND_URL || "http://localhost:8000";
const apiPaths = ["/sessions", "/baseline", "/uploads", "/studies", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(apiPaths.map((p) => [p, { target: backend, changeOrigin: true }])),
  },
});