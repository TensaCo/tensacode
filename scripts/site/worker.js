// The tensorcode.dev Worker (wrangler.jsonc). Every page is a static asset in site/;
// this script only normalizes the address before handing the request to the assets:
//
//   http://tensorcode.dev/...        -> 301 https://tensorcode.dev/...
//   http(s)://www.tensorcode.dev/... -> 301 https://tensorcode.dev/...
//
// Anything else (the workers.dev address, `wrangler dev`) is served as is, with the
// 404 page and trailing-slash rules configured under "assets".

const APEX = 'tensorcode.dev';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const www = url.hostname === `www.${APEX}`;
    if (www || (url.hostname === APEX && url.protocol === 'http:')) {
      url.protocol = 'https:';
      url.hostname = APEX;
      url.port = '';
      return new Response(null, {
        status: 301,
        headers: { location: url.toString(), 'cache-control': 'public, max-age=86400' },
      });
    }
    return env.ASSETS.fetch(request);
  },
};
