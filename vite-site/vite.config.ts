import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';
import { defineConfig, loadEnv } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
    plugins: [react()],
    server: {
      port: 7100,
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    define: {
      // Make environment variables available to client code
      __VITE_BACKEND_URL__: JSON.stringify(env.VITE_BACKEND_URL),
      __VITE_IS_MOCK__: JSON.stringify(env.VITE_IS_MOCK),
    },
  };
});
