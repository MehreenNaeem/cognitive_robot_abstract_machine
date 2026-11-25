/* ============================================================================
 * core/interaction-overlay.js — the interaction graph drawn into the 3D scene.
 *
 * The bridge serves /interaction: the selected objects of a task (where they stand,
 * what they afford) and the spatial relations between them. This module turns that
 * payload into marker-like build instructions (object names, curves, arrow heads
 * and captions in world coordinates) and matches a clicked mesh to its object, so the
 * geometry and the matching are testable without THREE.
 * ==========================================================================*/
(function (global) {
  'use strict';

  // %% looks

  const NODE_COLOR = '#fbbf3a';
  /* Colour of every object's name, the Interaction tab's node colour. */

  const RELATION_COLORS = {
    Support: '#4ade80',
    Containment: '#60a5fa',
    Contact: '#f472b6',
    Near: '#a78bfa',
  };
  /* Colour of each relation's curve, arrow head and caption. */

  const OTHER_RELATION_COLOR = '#dddddd';
  /* Colour of a relation missing from RELATION_COLORS. */

  const CAPTION_TEXT_COLOR = '#000000';
  /* Colour of the text of every relation caption. */

  const TIP_GAP = 0.035;
  /* Distance, in metres, a relation's arrow tip stops short of its target's centre. */

  const NAME_LIFT = 0.12;
  /* Height, in metres, of an object's name above where it stands. */

  const CURVE_LIFT = 0.25;
  /* Height, in metres, a relation's curve rises above the straight line at its middle. */

  const BEND_STEP = 0.12;
  /* Sideways bend, in metres, of each further relation between the same two objects. */

  const CURVE_SEGMENTS = 24;
  /* Number of straight pieces each relation's curve is drawn with. */

  const HEAD_LENGTH = 0.07;
  /* Length, in metres, of the arrow head at a relation's target. */

  const HEAD_WIDTH = 0.025;
  /* Diameter, in metres, of the arrow head's base. */

  const IDENTITY_ORIENTATION = [0, 0, 0, 1];
  /* A quaternion that does not rotate. */

  // %% vector helpers

  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function scaled(a, factor) { return [a[0] * factor, a[1] * factor, a[2] * factor]; }
  function length(a) { return Math.hypot(a[0], a[1], a[2]); }
  function normalized(a) {
    const size = length(a);
    return size === 0 ? [0, 0, 0] : scaled(a, 1 / size);
  }

  /* The horizontal unit vector to the right of the direction from `from` to `to`,
     or +X when the two stand one above the other. */
  function sideways(from, to) {
    const direction = subtract(to, from);
    const right = normalized([direction[1], -direction[0], 0]);
    return length(right) === 0 ? [1, 0, 0] : right;
  }

  /* The quaternion [x, y, z, w] turning +X onto `direction`. */
  function orientationAlong(direction) {
    const unit = normalized(direction);
    const dot = unit[0];
    if (dot < -0.999999) return [0, 0, 1, 0];          // opposite: half a turn about Z
    const axis = [0, -unit[2], unit[1]];               // +X crossed with `unit`
    const w = 1 + dot;
    const size = Math.hypot(axis[0], axis[1], axis[2], w);
    return [axis[0] / size, axis[1] / size, axis[2] / size, w / size];
  }

  // %% geometry

  /* The points of the curve from `from` to `to`, rising over its middle and bent
     sideways by `bend` metres. */
  function curvePoints(from, to, bend) {
    const middle = scaled(add(from, to), 0.5);
    const control = add(add(middle, [0, 0, CURVE_LIFT]), scaled(sideways(from, to), bend));
    const points = [];
    for (let index = 0; index <= CURVE_SEGMENTS; index++) {
      const t = index / CURVE_SEGMENTS;
      points.push(add(add(scaled(from, (1 - t) * (1 - t)), scaled(control, 2 * (1 - t) * t)),
        scaled(to, t * t)));
    }
    return points;
  }

  /* The point halfway along a curve's points. */
  function middleOf(points) {
    return points[Math.floor(points.length / 2)];
  }

  function relationColor(relation) {
    return RELATION_COLORS[relation] || OTHER_RELATION_COLOR;
  }

  /* The bend of every edge: each further edge between the same two objects, in
     either direction, bends one BEND_STEP further. */
  function bends(edges) {
    const seen = {};
    return edges.map(function (edge) {
      const pair = [edge.from, edge.to].sort().join('\u0000');
      seen[pair] = (seen[pair] || 0) + 1;
      return seen[pair] * BEND_STEP;
    });
  }

  function nameMarker(node) {
    return {
      kind: 'text', node: node.id, color: NODE_COLOR, text: node.label,
      pose: add(node.position, [0, 0, NAME_LIFT]).concat(IDENTITY_ORIENTATION),
    };
  }

  function edgeMarkers(edge, from, to, bend) {
    const color = relationColor(edge.label);
    const points = curvePoints(from.position, to.position, bend);
    const end = points[points.length - 1];
    const direction = normalized(subtract(end, points[points.length - 2]));
    const tip = subtract(end, scaled(direction, TIP_GAP));
    return [
      { kind: 'line_strip', color: color, points: points, pose: [0, 0, 0].concat(IDENTITY_ORIENTATION) },
      {
        kind: 'arrow', color: color,
        pose: subtract(tip, scaled(direction, HEAD_LENGTH)).concat(orientationAlong(direction)),
        scale: [HEAD_LENGTH, HEAD_WIDTH / 2, HEAD_WIDTH / 2],
      },
      {
        kind: 'text', caption: true, color: color, textColor: CAPTION_TEXT_COLOR,
        text: edge.label, pose: middleOf(points).concat(IDENTITY_ORIENTATION),
      },
    ];
  }

  /* Every build instruction of an /interaction payload: per object its name, per relation a curve, an arrow head at its target and its caption. An
     object's instructions carry its id as `node`; a relation's caption is marked
     `caption`, its text in `textColor` on a frame in the relation's `color`. */
  function markersOf(payload) {
    const nodes = (payload && payload.nodes) || [];
    const placed = {};
    nodes.forEach(function (node) { if (node.position) placed[node.id] = node; });
    const edges = ((payload && payload.edges) || []).filter(function (edge) {
      return placed[edge.from] && placed[edge.to];
    });
    const edgeBends = bends(edges);
    let markers = [];
    nodes.forEach(function (node) {
      if (placed[node.id]) markers.push(nameMarker(node));
    });
    edges.forEach(function (edge, index) {
      markers = markers.concat(edgeMarkers(edge, placed[edge.from], placed[edge.to], edgeBends[index]));
    });
    return markers;
  }

  /* What decides the drawing: the graph and where its objects stand. */
  function drawKey(payload) {
    const nodes = (payload && payload.nodes) || [];
    return ((payload && payload.signature) || '') + '#' + JSON.stringify(
      nodes.map(function (node) { return node.position || null; }));
  }

  // %% clicks

  /* The first object of the payload named by one of `names` (nearest first), a
     prefixed name ("apartment/drawer") matching by the part after its last slash. */
  function nodeFor(payload, names) {
    const byId = {};
    ((payload && payload.nodes) || []).forEach(function (node) { byId[node.id] = node; });
    for (let index = 0; index < (names || []).length; index++) {
      const name = String(names[index]);
      if (byId[name]) return byId[name];
      const bare = name.slice(name.lastIndexOf('/') + 1);
      if (byId[bare]) return byId[bare];
    }
    return null;
  }

  /* The card shown for a clicked object: its name over its description lines. */
  function cardOf(node) {
    return { title: node.label, lines: (node.lines || []).slice() };
  }

  global.InteractionOverlay = {
    NODE_COLOR: NODE_COLOR,
    RELATION_COLORS: RELATION_COLORS,
    OTHER_RELATION_COLOR: OTHER_RELATION_COLOR,
    CAPTION_TEXT_COLOR: CAPTION_TEXT_COLOR,
    BEND_STEP: BEND_STEP,
    CURVE_LIFT: CURVE_LIFT,
    curvePoints: curvePoints,
    orientationAlong: orientationAlong,
    markersOf: markersOf,
    drawKey: drawKey,
    nodeFor: nodeFor,
    cardOf: cardOf,
  };
})(typeof window !== 'undefined' ? window : this);
