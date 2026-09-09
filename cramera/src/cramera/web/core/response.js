/* ============================================================================
 * core/response.js — turns a fetch Response into parsed JSON or a clean error.
 *
 * Two different failures answer with a non-2xx status, and a panel has to tell them
 * apart. An endpoint that refused the request explains itself in the body
 * ({ok:false, error}) and that explanation is what the user needs to read. A host with
 * no matching backend route (a static site, or a stale reverse proxy) answers with an
 * HTML error page instead; parsing that as JSON throws a raw "SyntaxError: JSON.parse:
 * unexpected character…" that tells a user nothing, so it becomes "no server for this
 * route".
 * ==========================================================================*/
(function () {
  'use strict';

  async function parseJson(response) {
    const body = await readJson(response);
    if (body === null) throw new Error('no server for this route (HTTP ' + response.status + ')');
    if (response.ok) return body;
    throw new Error(body.error || 'no server for this route (HTTP ' + response.status + ')');
  }

  /* The response's JSON body, or null when it carries none (an HTML error page). */
  async function readJson(response) {
    try {
      const body = await response.json();
      return body && typeof body === 'object' ? body : null;
    } catch (error) {
      return null;
    }
  }

  window.ResponseUtil = { parseJson: parseJson };
})();
