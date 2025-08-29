/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        neon: {
          green: '#39ff14',
          cyan: '#00fff5',
        }
      }
    },
  },
  plugins: [],
}


