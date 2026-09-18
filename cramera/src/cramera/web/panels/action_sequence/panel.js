/* ============================================================================
 * panels/action_sequence/panel.js — grounding a task into an action sequence.
 *
 * A task statement ("Put the bowl in the drawer") goes to the running demo's bridge
 * (POST /action_sequence); the answer is the ordered actions a language model proposes
 * for it, checked against the objects that demo's world actually holds. Only the demo
 * process has that world, so the panel needs one running: without it there is nothing
 * to ground a task in, and the panel says so instead of offering a button that cannot
 * work.
 *
 * Bus events:
 *   listens  live:changed {on, url}   the bridge the scene panel found
 * ==========================================================================*/
Panels.define('action-sequence', function (root, bus) {
  root.innerHTML =
    '<div class="panel-head">' +
    '  <h2>Action sequence · task grounding</h2>' +
    '  <span id="sequence-status" class="knowledge-status">looking for a running demo…</span>' +
    '</div>' +
    '<div class="query-box">' +
    '  <div class="query-row">' +
    '    <textarea id="task-input" rows="2" spellcheck="false" placeholder="Put the bowl in the drawer"></textarea>' +
    '    <button id="task-run">Predict</button>' +
    '  </div>' +
    '  <div id="task-presets" class="presets"></div>' +
    '</div>' +
    '<div id="sequence" class="answer"></div>';

  const statusEl = root.querySelector('#sequence-status');
  const input = root.querySelector('#task-input');
  const runBtn = root.querySelector('#task-run');
  const presetsEl = root.querySelector('#task-presets');
  const answerEl = root.querySelector('#sequence');

  //: task statements to try without typing one
  const PRESETS = [
    'Put the bowl in the drawer',
    'Open the drawer',
    'Put the milk on the countertop',
  ];

  //: how often the panel asks whether a demo is reachable, in ms
  const PROBE_INTERVAL_MS = 2000;

  const ATTACHED = 'demo attached';
  const NO_DEMO = 'no running demo';
  const ASKING = 'asking the language model…';

  let bridge = null;      // url of the reachable bridge, or null while none answers
  let running = false;

  welcome();
  buildPresets();
  probe();
  setInterval(probe, PROBE_INTERVAL_MS);

  function welcome() {
    answerEl.innerHTML =
      '<div class="goal">action sequence · task grounding</div>' +
      '<p class="headline">A <b>task statement</b> is grounded into the <b>sequence of ' +
      'actions</b> that carries it out: each step names the <b>action</b>, the object it ' +
      'is performed <b>on</b> and the object it is performed <b>against</b>.</p>' +
      '<p class="hint-txt">The objects are the ones the running demo’s world holds, and ' +
      'every proposed step is checked against what its object can actually take part in — ' +
      'a step no object affords is rejected and asked for again.</p>';
  }

  // %% is a demo running?
  function probe() {
    if (running) return;              // a prediction is in flight; leave the status alone
    fetch(LiveMode.bridgeUrl() + '/info')
      .then(function (response) { return response.json(); })
      .then(function (info) { setBridge(info ? LiveMode.bridgeUrl() : null); })
      .catch(function () { setBridge(null); });
  }

  function setBridge(url) {
    bridge = url;
    statusEl.textContent = url ? ATTACHED : NO_DEMO;
    statusEl.classList.toggle('ready', !!url);
    runBtn.disabled = !url;
  }

  bus.on('live:changed', function (p) { if (p.url) setBridge(p.url); });

  // %% presets
  function buildPresets() {
    PRESETS.forEach(function (task) {
      const chip = document.createElement('div');
      chip.className = 'preset';
      chip.textContent = task;
      chip.addEventListener('click', function () {
        input.value = task;
        predict(task);
      });
      presetsEl.appendChild(chip);
    });
  }

  // %% ask the demo
  async function predict(task) {
    task = (task || '').trim();
    if (!task || running) return;
    if (!bridge) {
      render(task, { ok: false, error: 'no running demo to ground the task in' });
      return;
    }
    running = true;
    runBtn.textContent = '…';
    statusEl.textContent = ASKING;
    answerEl.innerHTML = '<div class="goal">&gt;&gt;&gt; ' + esc(task) + '</div>' +
      '<div class="nores">Grounding the task — this waits on the language model.</div>';
    try {
      const response = await fetch(bridge + '/action_sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: task }),
      });
      render(task, await ResponseUtil.parseJson(response));
    } catch (err) {
      render(task, { ok: false, error: errorText(err) });
    }
    running = false;
    runBtn.textContent = 'Predict';
    statusEl.textContent = bridge ? ATTACHED : NO_DEMO;
  }

  function render(task, result) {
    let html = '<div class="goal">&gt;&gt;&gt; ' + esc(task) + '</div>';
    if (!result.ok) {
      answerEl.innerHTML = html + '<div class="qerr">' + esc(result.error || 'prediction failed') + '</div>';
      return;
    }
    const actions = result.actions || [];
    if (!actions.length) {
      answerEl.innerHTML = html + '<div class="nores">No actions — the task grounded to an empty sequence.</div>';
      return;
    }
    html += '<p class="headline"><b>' + actions.length + '</b> action' +
      (actions.length === 1 ? '' : 's') + ' to perform, in order.</p>';
    actions.forEach(function (action, index) {
      html += stepRow(action, index + 1);
    });
    if (result.objects && result.objects.length) {
      html += '<div class="ansub">Objects in the demo</div>' +
        '<div class="ansrow"><div class="body"><span class="sub">' +
        esc(result.objects.join('  ·  ')) + '</span></div></div>';
    }
    answerEl.innerHTML = html;
  }

  function stepRow(action, step) {
    return '<div class="ansrow"><span class="tag step">' + step + '</span>' +
      '<div class="body"><span class="name">' + esc(action.action) + '</span>' +
      '<span class="sub">' + esc(action.sourceObject) + ' → ' + esc(action.targetObject) +
      '</span></div></div>';
  }

  runBtn.addEventListener('click', function () { predict(input.value); });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); predict(input.value); }
  });

  // an Error carries the useful text on .message; anything else is shown as-is
  function errorText(err) { return String((err && err.message) || err); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
});
