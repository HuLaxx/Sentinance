import type { Config } from "tailwindcss";

// Signal Atlas palette. The generic zinc/slate kit is remapped to the atlas inks, and the
// blue/sky/indigo/purple accents are remapped to copper, so every page shares one identity.
// Emerald, red, rose and amber are left alone: they carry price direction and warnings.
const ink = {
  50: "#f7f4ec",
  100: "#f2eee3",
  200: "#dedbd0",
  300: "#c4c9bf",
  400: "#abb3a7",
  500: "#858d81",
  600: "#5f675c",
  700: "#484d45",
  800: "#373d38",
  900: "#202622",
  950: "#191e1c",
};

const copper = {
  50: "#fbf1e8",
  100: "#f6e0cc",
  200: "#efc7a5",
  300: "#e8b285",
  400: "#e4a16b",
  500: "#d0864d",
  600: "#a45a32",
  700: "#91502f",
  800: "#603c26",
  900: "#3f2818",
  950: "#2a1a10",
};

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        zinc: ink,
        slate: ink,
        gray: ink,
        neutral: ink,
        sky: copper,
        cyan: copper,
        blue: copper,
        indigo: copper,
        violet: copper,
        purple: copper,
      },
      borderRadius: {
        lg: "3px",
        xl: "3px",
        "2xl": "4px",
        "3xl": "4px",
      },
      fontFamily: {
        sans: ["var(--font-plex-sans)", "IBM Plex Sans", "Arial", "sans-serif"],
        serif: ["var(--font-hedvig)", "Hedvig Letters Serif", "Georgia", "serif"],
        mono: ["var(--font-plex-mono)", "IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
