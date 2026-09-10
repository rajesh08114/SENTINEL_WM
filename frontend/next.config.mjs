/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",              // static bundle -> nginx; no Node server in prod
  reactStrictMode: true,
  trailingSlash: true,
  images: { unoptimized: true },
  eslint: { ignoreDuringBuilds: true },
};
export default nextConfig;
