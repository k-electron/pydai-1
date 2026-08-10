import os from 'node:os';

/** @type {import('next').NextConfig} */
const backend = process.env.EDGAR_API_URL ?? 'http://127.0.0.1:8000';

/**
 * Hosts allowed to request dev-server assets.
 *
 * Next 16 only trusts `localhost` by default and refuses its own JS chunks to anything
 * else, which loads the page HTML but leaves the UI dead. Reaching the dev server by
 * `127.0.0.1` or by the machine's LAN address (to try it from a phone, say) both trip it.
 *
 * Detected rather than hardcoded: writing one machine's IP into the repo would break the
 * next person exactly the way it is broken now. Set EDGAR_DEV_ORIGINS (comma-separated)
 * to add more.
 */
function localAddresses() {
  const found = new Set(['localhost', '127.0.0.1', '::1']);
  try {
    for (const addresses of Object.values(os.networkInterfaces())) {
      for (const address of addresses ?? []) {
        if (!address.internal) found.add(address.address);
      }
    }
  } catch {
    // `uv_interface_addresses` fails outright in some sandboxes and hardened CI
    // runners. This runs at config load, so an uncaught throw here fails the whole
    // build over a dev-only convenience; loopback alone is a fine fallback.
  }
  for (const extra of (process.env.EDGAR_DEV_ORIGINS ?? '').split(',')) {
    const trimmed = extra.trim();
    if (trimmed) found.add(trimmed);
  }
  return [...found];
}

const nextConfig = {
  allowedDevOrigins: localAddresses(),

  // Proxy the API through Next so the browser sees one origin. Without this the
  // streaming response is a cross-origin request and needs CORS on every hop.
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
