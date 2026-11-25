// Unit tests for web/core/interaction-overlay.js (node:test): the interaction graph
// drawn into the 3D scene, and a clicked mesh matched to its object.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/interaction-overlay.js'), 'utf8'))(scope);
  return scope.InteractionOverlay;
}

const MILK = { id: 'milk.stl', label: 'milk.stl', lines: ['Type: Milk'], position: [1, 0, 1] };
const TABLE = { id: 'table', label: 'table', lines: ['Type: Table'], position: [0, 0, 0.5] };

function payload(edges) {
  return { signature: 'graph', nodes: [MILK, TABLE], edges: edges || [] };
}

function close(actual, expected) {
  actual.forEach(function (value, index) {
    assert.ok(Math.abs(value - expected[index]) < 1e-9, actual + ' is not ' + expected);
  });
}

// %% objects

test('no object is drawn as a sphere', function () {
  const overlay = load();
  const markers = overlay.markersOf(payload([{ from: MILK.id, to: TABLE.id, label: 'Support' }]));
  assert.deepStrictEqual(markers.filter(function (m) { return m.kind === 'sphere'; }), []);
});

test('every object is named above where it stands, tagged with its id', function () {
  const overlay = load();
  const names = overlay.markersOf(payload()).filter(function (m) { return m.kind === 'text'; });
  assert.deepStrictEqual(names.map(function (m) { return m.text; }), [MILK.label, TABLE.label]);
  assert.deepStrictEqual(names.map(function (m) { return m.node; }), [MILK.id, TABLE.id]);
  assert.ok(names.every(function (m) { return m.color === overlay.NODE_COLOR; }));
  names.forEach(function (m, index) {
    const position = [MILK, TABLE][index].position;
    assert.deepStrictEqual(m.pose.slice(0, 2), position.slice(0, 2));
    assert.ok(m.pose[2] > position[2]);
  });
});

test('an object without a position is not drawn', function () {
  const overlay = load();
  const unplaced = { signature: 'g', nodes: [{ id: 'x', label: 'x', lines: [] }], edges: [] };
  assert.deepStrictEqual(overlay.markersOf(unplaced), []);
});

// %% relations

test('a relation is a curve from its source to its target in its colour', function () {
  const overlay = load();
  const curves = overlay.markersOf(payload([{ from: MILK.id, to: TABLE.id, label: 'Support' }]))
    .filter(function (m) { return m.kind === 'line_strip'; });
  assert.strictEqual(curves.length, 1);
  assert.deepStrictEqual(curves[0].points[0], MILK.position);
  assert.deepStrictEqual(curves[0].points[curves[0].points.length - 1], TABLE.position);
  assert.strictEqual(curves[0].color, overlay.RELATION_COLORS.Support);
});

test('a relation is captioned with its name', function () {
  const overlay = load();
  const captions = overlay.markersOf(payload([{ from: MILK.id, to: TABLE.id, label: 'Near' }]))
    .filter(function (m) { return m.kind === 'text' && m.node === undefined; });
  assert.deepStrictEqual(captions.map(function (m) { return m.text; }), ['Near']);
});

test('a relation caption is written in black without the object dot', function () {
  const overlay = load();
  const caption = overlay.markersOf(payload([{ from: MILK.id, to: TABLE.id, label: 'Support' }]))
    .find(function (m) { return m.caption; });
  assert.strictEqual(caption.text, 'Support');
  assert.strictEqual(caption.textColor, overlay.CAPTION_TEXT_COLOR);
  assert.strictEqual(caption.color, overlay.RELATION_COLORS.Support);
});

test('a relation caption sits on the middle of its own curve', function () {
  const overlay = load();
  const captions = overlay.markersOf(payload([
    { from: MILK.id, to: TABLE.id, label: 'Support' },
    { from: MILK.id, to: TABLE.id, label: 'Near' },
  ])).filter(function (m) { return m.caption; });
  [1, 2].forEach(function (step, index) {
    const points = overlay.curvePoints(MILK.position, TABLE.position, step * overlay.BEND_STEP);
    close(captions[index].pose.slice(0, 3), points[Math.floor(points.length / 2)]);
  });
});

test('an object name is not a relation caption', function () {
  const overlay = load();
  const names = overlay.markersOf(payload()).filter(function (m) { return m.kind === 'text'; });
  assert.ok(names.every(function (m) { return !m.caption; }));
});

test('a relation ends in an arrow head pointing at its target', function () {
  const overlay = load();
  const heads = overlay.markersOf(payload([{ from: MILK.id, to: TABLE.id, label: 'Contact' }]))
    .filter(function (m) { return m.kind === 'arrow'; });
  assert.strictEqual(heads.length, 1);
  assert.ok(Math.hypot(heads[0].pose[0] - TABLE.position[0], heads[0].pose[1] - TABLE.position[1],
    heads[0].pose[2] - TABLE.position[2]) < 0.2);
});

test('an unknown relation is drawn in the fallback colour', function () {
  const overlay = load();
  const curve = overlay.markersOf(payload([{ from: MILK.id, to: TABLE.id, label: 'Other' }]))
    .find(function (m) { return m.kind === 'line_strip'; });
  assert.strictEqual(curve.color, overlay.OTHER_RELATION_COLOR);
});

test('a relation to an object without a position is not drawn', function () {
  const overlay = load();
  const markers = overlay.markersOf({
    signature: 'g', nodes: [MILK, { id: 'x', label: 'x', lines: [] }],
    edges: [{ from: MILK.id, to: 'x', label: 'Near' }],
  });
  assert.deepStrictEqual(markers.filter(function (m) { return m.kind === 'line_strip'; }), []);
});

test('further relations between the same objects bend further apart', function () {
  const overlay = load();
  const curves = overlay.markersOf(payload([
    { from: MILK.id, to: TABLE.id, label: 'Support' },
    { from: MILK.id, to: TABLE.id, label: 'Near' },
  ])).filter(function (m) { return m.kind === 'line_strip'; });
  close(curves[0].points[1], overlay.curvePoints(MILK.position, TABLE.position,
    overlay.BEND_STEP)[1]);
  close(curves[1].points[1], overlay.curvePoints(MILK.position, TABLE.position,
    2 * overlay.BEND_STEP)[1]);
});

test('a curve rises above the straight line at its middle', function () {
  const overlay = load();
  const points = overlay.curvePoints([0, 0, 0], [2, 0, 0], 0);
  const middle = points[Math.floor(points.length / 2)];
  close(middle, [1, 0, overlay.CURVE_LIFT / 2]);
});

test('the arrow orientation turns +X onto the direction', function () {
  const overlay = load();
  close(overlay.orientationAlong([1, 0, 0]), [0, 0, 0, 1]);
  close(overlay.orientationAlong([0, 1, 0]), [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
  close(overlay.orientationAlong([-1, 0, 0]), [0, 0, 1, 0]);
});

// %% redrawing

test('moving an object changes the draw key', function () {
  const overlay = load();
  const moved = { signature: 'graph', nodes: [Object.assign({}, MILK, { position: [2, 0, 1] }), TABLE], edges: [] };
  assert.notStrictEqual(overlay.drawKey(payload()), overlay.drawKey(moved));
});

test('the same graph has the same draw key', function () {
  const overlay = load();
  assert.strictEqual(overlay.drawKey(payload()), overlay.drawKey(payload()));
});

// %% clicks

test('a clicked name finds its object', function () {
  const overlay = load();
  assert.strictEqual(overlay.nodeFor(payload(), ['wall', 'table']), TABLE);
});

test('the nearest matching name wins', function () {
  const overlay = load();
  assert.strictEqual(overlay.nodeFor(payload(), ['milk.stl', 'table']), MILK);
});

test('a prefixed link name finds its object', function () {
  const overlay = load();
  assert.strictEqual(overlay.nodeFor(payload(), ['apartment/table']), TABLE);
});

test('a name outside the graph finds nothing', function () {
  const overlay = load();
  assert.strictEqual(overlay.nodeFor(payload(), ['wall']), null);
});

test('the card shows the object name over its description', function () {
  const overlay = load();
  assert.deepStrictEqual(overlay.cardOf(MILK), { title: MILK.label, lines: MILK.lines });
});
