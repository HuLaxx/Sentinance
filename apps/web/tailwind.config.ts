import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)"],
        script: ["var(--font-script)"],
      },
    },
  },
  plugins: [],
};

export default config;
