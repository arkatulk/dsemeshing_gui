import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';


export default defineConfig({
  plugins: [react(),tailwindcss(),],
  server: {
    proxy: {
      "/upload": "http://127.0.0.1:8000",
      "/download": "http://127.0.0.1:8000",
    },
  },
});
