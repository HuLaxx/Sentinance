/** @type {import("next").NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Self-contained server bundle copied by apps/web/Dockerfile (.next/standalone/server.js).
  output: 'standalone',
};

export default nextConfig;
