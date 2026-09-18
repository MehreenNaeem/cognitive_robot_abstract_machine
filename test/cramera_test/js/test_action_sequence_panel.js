// Unit tests for panels/action_sequence/panel.js (node:test): what the task-grounding
// panel asks the running demo, and what it renders from the answer.
//
// As in test_graph_panel.js, the panel's free variables (Panels, LiveMode, fetch,
// ResponseUtil, document, setInterval) are bound as explicit function parameters. LiveMode
// and ResponseUtil are the real core files, so the bridge url and the error handling are
// exercised rather than stubbed.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'panels/action_sequence/panel.js'), 'utf8');

const BRIDGE = 'http://localhost:8765';

function loadCore(file, scope) {
  new Function('window', fs.readFileSync(path.join(WEB, file), 'utf8'))(scope);
  return scope;
}

function flush() {
  return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

// %% stubs of the interfaces panel.js reads
function makeElement() {
  const element = {
    innerHTML: '',
    textContent: '',
    className: '',
    disabled: false,
    value: '',
    children: [],
    classList: { toggle() {}, add() {}, remove() {} },
    handlers: {},
    addEventListener(event, callback) { element.handlers[event] = callback; },
    appendChild(child) { element.children.push(child); },
  };
  return element;
}

function makeRoot() {
  const byId = {
    '#sequence-status': makeElement(),
    '#task-input': makeElement(),
    '#task-run': makeElement(),
    '#task-presets': makeElement(),
    '#sequence': makeElement(),
  };
  return {
    innerHTML: '',
    querySelector(selector) { return byId[selector]; },
    control(selector) { return byId[selector]; },
  };
}

function makeBus() {
  const handlers = {};
  return {
    on(event, callback) { (handlers[event] = handlers[event] || []).push(callback); },
    emit(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
  };
}

// records every request; answers from `responses`, each {body, status?}, keyed by url
function makeFetch(responses, requests) {
  return async function fetch(url, options) {
    requests.push({ url: url, options: options });
    const answer = responses[url];
    if (answer === undefined) throw new Error('no demo there');
    const status = answer.status || 200;
    return {
      ok: status < 400,
      status: status,
      json: async function () { return answer.body; },
    };
  };
}

function loadPanel(responses) {
  const requests = [];
  let factory = null;
  const Panels = { define(id, f) { factory = f; } };
  const scope = { location: { hostname: 'localhost', search: '' } };
  loadCore('core/live-mode.js', scope);
  loadCore('core/response.js', scope);
  const document = { createElement() { return makeElement(); } };
  new Function('Panels', 'LiveMode', 'fetch', 'ResponseUtil', 'document', 'setInterval', SOURCE)(
    Panels, scope.LiveMode, makeFetch(responses, requests), scope.ResponseUtil, document,
    function () {}
  );
  const root = makeRoot();
  factory(root, makeBus());
  return { root: root, requests: requests };
}

function predict(panel, task) {
  panel.root.control('#task-input').value = task;
  panel.root.control('#task-run').handlers.click();
  return flush();
}

function asked(panel) {
  return panel.requests.filter(function (request) {
    return request.url === BRIDGE + '/action_sequence';
  });
}

const RUNNING_DEMO = { body: { running: true } };
const SEQUENCE = {
  body: {
    ok: true,
    task: 'Put the bowl in the drawer',
    objects: ['Bowl', 'Drawer'],
    actions: [
      { action: 'PickUpAction', sourceObject: 'Bowl', targetObject: 'Drawer' },
      { action: 'PlaceAction', sourceObject: 'Bowl', targetObject: 'Drawer' },
    ],
  },
};

// %% finding the demo
test('the panel reports whether a demo is running', async function () {
  const panel = loadPanel({ [BRIDGE + '/info']: RUNNING_DEMO });
  await flush();

  assert.strictEqual(panel.root.control('#sequence-status').textContent, 'demo attached');
  assert.strictEqual(panel.root.control('#task-run').disabled, false);
});

test('without a demo the panel says so and asks nobody', async function () {
  const panel = loadPanel({});
  await flush();

  assert.strictEqual(panel.root.control('#sequence-status').textContent, 'no running demo');
  assert.strictEqual(panel.root.control('#task-run').disabled, true);

  await predict(panel, 'Put the bowl in the drawer');

  assert.match(panel.root.control('#sequence').innerHTML, /no running demo/);
  assert.deepStrictEqual(asked(panel), []);
});

// %% asking the demo
test('the task statement is posted to the running demo', async function () {
  const panel = loadPanel({
    [BRIDGE + '/info']: RUNNING_DEMO,
    [BRIDGE + '/action_sequence']: SEQUENCE,
  });
  await flush();

  await predict(panel, 'Put the bowl in the drawer');

  const requests = asked(panel);
  assert.strictEqual(requests.length, 1);
  assert.strictEqual(requests[0].options.method, 'POST');
  assert.deepStrictEqual(JSON.parse(requests[0].options.body), {
    task: 'Put the bowl in the drawer',
  });
});

test('the predicted actions are rendered in order, with the demo objects they were chosen from',
  async function () {
    const panel = loadPanel({
      [BRIDGE + '/info']: RUNNING_DEMO,
      [BRIDGE + '/action_sequence']: SEQUENCE,
    });
    await flush();

    await predict(panel, 'Put the bowl in the drawer');

    const rendered = panel.root.control('#sequence').innerHTML;
    assert.ok(rendered.indexOf('PickUpAction') < rendered.indexOf('PlaceAction'));
    assert.match(rendered, /Bowl → Drawer/);
    assert.match(rendered, /2<\/b> actions/);
    assert.match(rendered, /Bowl {2}· {2}Drawer/);
    assert.strictEqual(panel.root.control('#sequence-status').textContent, 'demo attached');
  });

test('a refused prediction is shown in the words the demo refused it with',
  async function () {
    /*
     * The bridge marks a refusal with an HTTP status and explains it in the body; the
     * explanation is the whole content of the message, so it must not be swallowed in
     * favour of the status.
     */
    const refusal = 'the language model at http://model could not be reached';
    const panel = loadPanel({
      [BRIDGE + '/info']: RUNNING_DEMO,
      [BRIDGE + '/action_sequence']: { status: 502, body: { ok: false, error: refusal } },
    });
    await flush();

    await predict(panel, 'Fly to the moon');

    const rendered = panel.root.control('#sequence').innerHTML;
    assert.match(rendered, /could not be reached/);
    assert.doesNotMatch(rendered, /no server for this route/);
  });

test('a demo that goes away mid-request is shown as the failure it is', async function () {
  const panel = loadPanel({ [BRIDGE + '/info']: RUNNING_DEMO });
  await flush();

  await predict(panel, 'Put the bowl in the drawer');

  assert.match(panel.root.control('#sequence').innerHTML, /no demo there/);
});

test('an empty task statement is not sent anywhere', async function () {
  const panel = loadPanel({
    [BRIDGE + '/info']: RUNNING_DEMO,
    [BRIDGE + '/action_sequence']: SEQUENCE,
  });
  await flush();

  await predict(panel, '   ');

  assert.deepStrictEqual(asked(panel), []);
});
