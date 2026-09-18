/* ============================================================================
 * panels/llm_planner/panel.js — the LLM planner's tree.
 *
 * Three levels, from the task down to the robot: the task statement a language model
 * was given, the actions it grounded the task into, and the plan steps each of those
 * is performed as (GET /llm_plan on the running demo's bridge). Every node carries the
 * status its plan reports, so a step colours as it runs.
 *
 * Drawn as nested rows rather than through the graph panel's network, which is a single
 * shared instance and cannot be mounted twice. The tree outlives the demo that published
 * it, so a finished run stays on screen.
 * ==========================================================================*/
Panels.define('llm-planner', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>LLM Planner · task to plan</h2>' +
    '  <span id="llm-plan-status" class="knowledge-status">looking for a running demo…</span>' +
    '</div>' +
    '<div id="llm-plan" class="llm-plan"></div>';

  const statusEl = root.querySelector('#llm-plan-status');
  const treeEl = root.querySelector('#llm-plan');

  //: how often the tree is refetched, in ms — often enough to follow a running step
  const POLL_INTERVAL_MS = 500;

  const ATTACHED = 'demo attached';
  const NO_DEMO = 'no running demo';

  let drawn = '';          // what is on screen, so an unchanged tree is left alone

  empty('Waiting for a task — the tree appears once a demo grounds one.');
  poll();
  setInterval(poll, POLL_INTERVAL_MS);

  function empty(message) {
    treeEl.innerHTML = '<div class="answer-empty">' + esc(message) + '</div>';
  }

  // %% the running demo
  function poll() {
    fetch(LiveMode.bridgeUrl() + '/llm_plan')
      .then(function (response) { return response.json(); })
      .then(function (payload) {
        setStatus(true);
        render(payload);
      })
      .catch(function () {
        // the demo ending is the normal end of a run, and what it planned is still
        // worth reading: only the status says the demo is gone, the tree stays
        setStatus(false);
      });
  }

  function setStatus(attached) {
    statusEl.textContent = attached ? ATTACHED : NO_DEMO;
    statusEl.classList.toggle('ready', attached);
  }

  bus.on('live:changed', function (p) { if (p.on) setStatus(true); });

  // %% the tree
  function render(payload) {
    const nodes = (payload && payload.nodes) || [];
    if (!nodes.length) {
      if (drawn) { drawn = ''; empty('Waiting for a task — the tree appears once a demo grounds one.'); }
      return;
    }
    const state = JSON.stringify(nodes.map(function (node) {
      return [node.id, node.parent, node.label, node.status];
    }));
    if (state === drawn) return;               // same tree, same statuses
    drawn = state;
    treeEl.innerHTML = childrenOf(nodes, null).map(function (node) {
      return branch(nodes, node, 0);
    }).join('');
  }

  function childrenOf(nodes, parent) {
    return nodes.filter(function (node) { return node.parent === parent; });
  }

  function branch(nodes, node, depth) {
    const children = childrenOf(nodes, node.id);
    return row(node, depth) + children.map(function (child) {
      return branch(nodes, child, depth + 1);
    }).join('');
  }

  function row(node, depth) {
    return '<div class="llm-row ' + statusClass(node.status) + '" style="--depth:' + depth + '">' +
      '<span class="llm-dot"></span>' +
      '<span class="llm-label">' + esc(node.label) + '</span>' +
      '<span class="llm-status">' + esc(String(node.status || '').toLowerCase()) + '</span>' +
      '</div>';
  }

  //: the statuses coraplex reports, as the classes app.css colours them with
  const STATUS_CLASSES = {
    CREATED: 'created', RUNNING: 'running', SUCCEEDED: 'succeeded',
    FAILED: 'failed', INTERRUPTED: 'failed', PAUSE: 'paused',
  };
  function statusClass(status) {
    return 'st-' + (STATUS_CLASSES[status] || 'created');
  }

  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
});
