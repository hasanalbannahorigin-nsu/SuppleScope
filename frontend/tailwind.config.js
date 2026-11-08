/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#071427",
          900: "#0B1F3A",
          800: "#122B4A",
          700: "#1A3A5C",
        },
        teal: {
          400: "#2DD4BF",
          500: "#14B8A6",
          600: "#0D9488",
          700: "#0F766E",
        },
      },
      fontFamily: {
        sans: [
          '"Source Sans 3"',
          '"Noto Sans Bengali"',
          '"Hind Siliguri"',
          "system-ui",
          "sans-serif",
        ],
        bn: ['"Noto Sans Bengali"', '"Hind Siliguri"', "sans-serif"],
      },
      boxShadow: {
        card: "0 10px 40px -20px rgba(11, 31, 58, 0.45)",
      },
    },
  },
  plugins: [],
};
