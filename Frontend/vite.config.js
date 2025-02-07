import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
   plugins: [react()],
   css: {
      postcss: './postcss.config.js',
   },
   define: {
      'process.env': {},
   },
   server: {
      headers: {
         'Cross-Origin-Opener-Policy': 'same-origin',
         'Cross-Origin-Embedder-Policy': 'require-corp',
      },
      mimeTypes: {
         'video/mp4': ['mp4'],
      },
   },
});
