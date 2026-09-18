// Unit tests for panels/llm_planner/panel.js (node:test): the task-to-plan tree.
//
// The panel's free variables (Panels, LiveMode, fetch, setInterval) are bound as
// explicit function parameters, and LiveMode is the real core/live-mode.js so the
// bridge url is exercised rather than stubbed.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const SOURCE = fs.readFileSync(path.join(WEB, 'panels/llm_planner/panel.js'), 'utf8');

const BRIDGE = 'http://localhost:8765';
const ENDPOINT = BRIDGE + '/llm_plan';

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
    classList: { toggle() {}, add() {}, remove() {} },
    handlers: {},
    addEventListener(event, callback) { element.handlers[event] = callback; },
  };
  return element;
}

function makeRoot() {
  const byId = {
    '#llm-plan-status': makeElement(),
    '#llm-plan': makeElement(),
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

// `answers` is a queue: each poll takes the next one, the last one repeats
function loadPanel(answers) {
  const requested = [];
  let factory = null;
  const Panels = { define(id, f) { factory = f; } };
  const scope = { location: { hostname: 'localhost', search: '' } };
  loadCore('core/live-mode.js', scope);
  const queue = answers.slice();
  const fetch = async function (url) {
    requested.push(url);
    const answer = queue.length > 1 ? queue.shift() : queue[0];
    if (answer instanceof Error) throw answer;
    return { ok: true, status: 200, json: async function () { return answer; } };
  };
  let tick = null;
  new Function('Panels', 'LiveMode', 'fetch', 'setInterval', SOURCE)(
    Panels, scope.LiveMode, fetch, function (callback) { tick = callback; }
  );
  const root = makeRoot();
  factory(root, makeBus());
  return {
    root: root,
    requested: requested,
    poll: async function () { tick(); await flush(); },
  };
}

function node(id, parent, label, status) {
  return { id: id, parent: parent, label: label, status: status, group: 'action' };
}

const TASK_TREE = {
  task: 'Put the bowl in the drawer',
  signature: 'llm_task|llm_step_0|llm_step_0_plan_node_1',
  nodes: [
    node('llm_task', null, 'Put the bowl in the drawer', 'RUNNING'),
    node('llm_step_0', 'llm_task', 'PickUpAction', 'RUNNING'),
    node('llm_step_0_plan_node_1', 'llm_step_0', 'ParkArmsAction', 'SUCCEEDED'),
    node('llm_step_0_plan_node_2', 'llm_step_0', 'PickUpAction', 'RUNNING'),
  ],
};

const NO_PLAN = { task: '', signature: '', nodes: [] };

// %% where the tree comes from
test('the tree is read from the running demo', async function () {
  const panel = loadPanel([NO_PLAN]);
  await flush();

  assert.deepStrictEqual(panel.requested, [ENDPOINT]);
  assert.strictEqual(panel.root.control('#llm-plan-status').textContent, 'demo attached');
});

test('without a demo the panel says so', async function () {
  const panel = loadPanel([new Error('no demo there')]);
  await flush();

  assert.strictEqual(panel.root.control('#llm-plan-status').textContent, 'no running demo');
});

test('with no task grounded yet the panel waits', async function () {
  const panel = loadPanel([NO_PLAN]);
  await flush();

  assert.match(panel.root.control('#llm-plan').innerHTML, /Waiting for a task/);
});

// %% the tree itself
test('the task is drawn above the actions it was grounded into', async function () {
  const panel = loadPanel([TASK_TREE]);
  await flush();

  const drawn = panel.root.control('#llm-plan').innerHTML;
  assert.ok(drawn.indexOf('Put the bowl in the drawer') < drawn.indexOf('PickUpAction'));
  assert.match(drawn, /ParkArmsAction/);
});

test('each level is drawn one step deeper than its parent', async function () {
  const panel = loadPanel([TASK_TREE]);
  await flush();

  const depths = [];
  const pattern = /--depth:(\d+)/g;
  let match;
  while ((match = pattern.exec(panel.root.control('#llm-plan').innerHTML)) !== null) {
    depths.push(Number(match[1]));
  }
  assert.deepStrictEqual(depths, [0, 1, 2, 2]);
});

test('a step carries the status its plan reports', async function () {
  const panel = loadPanel([TASK_TREE]);
  await flush();

  const rows = panel.root.control('#llm-plan').innerHTML.split('<div class="llm-row ');
  const parkArms = rows.find(function (row) { return row.indexOf('ParkArmsAction') >= 0; });
  const pickUp = rows.find(function (row) { return row.indexOf('>PickUpAction<') >= 0; });
  assert.match(parkArms, /^st-succeeded/);
  assert.match(pickUp, /^st-running/);
});

// %% keeping up with the demo
test('a tree that has not changed is left alone', async function () {
  const panel = loadPanel([TASK_TREE]);
  await flush();
  const drawn = panel.root.control('#llm-plan').innerHTML;
  panel.root.control('#llm-plan').innerHTML = 'untouched';

  await panel.poll();

  assert.strictEqual(panel.root.control('#llm-plan').innerHTML, 'untouched');
  assert.match(drawn, /PickUpAction/);
});

test('a status change is redrawn', async function () {
  const finished = {
    task: TASK_TREE.task,
    signature: TASK_TREE.signature,
    nodes: TASK_TREE.nodes.map(function (n) {
      return node(n.id, n.parent, n.label, 'SUCCEEDED');
    }),
  };
  const panel = loadPanel([TASK_TREE, finished]);
  await flush();
  assert.match(panel.root.control('#llm-plan').innerHTML, /st-running/);

  await panel.poll();

  assert.doesNotMatch(panel.root.control('#llm-plan').innerHTML, /st-running/);
});

test('the tree outlives the demo that published it', async function () {
  /*
   * A demo ending is how a run finishes, and what it planned is what the user is
   * reading - so only the status chip changes, the tree stays.
   */
  const panel = loadPanel([TASK_TREE, new Error('no demo there')]);
  await flush();
  const drawn = panel.root.control('#llm-plan').innerHTML;

  await panel.poll();

  assert.strictEqual(panel.root.control('#llm-plan').innerHTML, drawn);
  assert.match(drawn, /PickUpAction/);
  assert.strictEqual(panel.root.control('#llm-plan-status').textContent, 'no running demo');
});
