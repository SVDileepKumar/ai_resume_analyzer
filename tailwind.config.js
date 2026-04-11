/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/templates/**/*.html",
    "./app/static/js/**/*.js",
  ],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        brand: {
          blue: "#0053e2",
          "blue-hover": "#0046c0",
          "blue-pressed": "#003a9e",
          accent: "#ffc220",
          "accent-dark": "#995213",
        },
        success: "#2a8703",
        danger: "#ea1100",
        warning: "#995213",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
