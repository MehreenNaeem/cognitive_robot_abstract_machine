// Unit tests for web/core/response.js (node:test).
// Two failures answer with a non-2xx status and must not be reported the same way: an
// endpoint that refused the request explains itself in the body, which is what the user
// needs to read, while a route with no backend (a static host, or a stale reverse proxy)
// answers with an HTML error page and must not surface a raw JSON.parse SyntaxError.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/response.js'), 'utf8'))();
}

function fakeResponse(ok, status, jsonPayload) {
  return {
    ok: ok,
    status: status,
    json: function () {
      return jsonPayload === undefined
        ? Promise.reject(new SyntaxError('JSON.parse: unexpected character at line 1 column 1 of the JSON data'))
        : Promise.resolve(jsonPayload);
    },
  };
}

test('parseJson() resolves the parsed body for an ok response', async function () {
  load();
  const payload = await window.ResponseUtil.parseJson(fakeResponse(true, 200, { ok: true }));
  assert.deepStrictEqual(payload, { ok: true });
});

test('parseJson() throws a clean error for a 404 instead of parsing its HTML body', async function () {
  load();
  await assert.rejects(
    window.ResponseUtil.parseJson(fakeResponse(false, 404)),
    /no server for this route \(HTTP 404\)/
  );
});

test('parseJson() throws a clean error for any non-ok status without a body', async function () {
  load();
  await assert.rejects(
    window.ResponseUtil.parseJson(fakeResponse(false, 500)),
    /no server for this route \(HTTP 500\)/
  );
});

test('parseJson() throws what a refusing endpoint said, not the generic message',
  async function () {
    /*
     * A refusal the server took the trouble to explain -- an unreachable language
     * model, a rejected query -- must reach the user in the server's own words; the
     * status alone would say nothing about what went wrong.
     */
    load();
    const refusal = fakeResponse(false, 502, {
      ok: false, error: 'the language model at http://model could not be reached',
    });

    await assert.rejects(
      window.ResponseUtil.parseJson(refusal),
      /the language model at http:\/\/model could not be reached/
    );
  });

test('parseJson() falls back to the generic message when a refusal explains nothing',
  async function () {
    load();
    await assert.rejects(
      window.ResponseUtil.parseJson(fakeResponse(false, 502, { ok: false })),
      /no server for this route \(HTTP 502\)/
    );
  });
