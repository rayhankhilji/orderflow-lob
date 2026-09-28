import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VITE_DEMO=1 builds the GitHub Pages static demo: API calls are served from
// baked fixtures (src/demo.js), base path is the project Pages URL, and the
// output lands in ../docs which Pages serves from main.
const demo = process.env.VITE_DEMO === "1";

export default defineConfig({
  plugins: [react()],
  base: demo ? "/orderflow-lob/" : "/",
  // emptyOutDir stays false: docs/ also hosts docs/figures/*.png referenced by
  // the README — hashed asset names make stale-file accumulation harmless.
  build: demo ? { outDir: "../docs", emptyOutDir: false } : {},
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
