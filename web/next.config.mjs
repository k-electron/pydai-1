/** @type {import('next').NextConfig} */
const backend = process.env.EDGAR_API_URL ?? 'http://127.0.0.1:8000';

const nextConfig = {
  // Proxy the API through Next so the browser sees one origin. Without this the
  // streaming response is a cross-origin request and needs CORS on every hop.
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
