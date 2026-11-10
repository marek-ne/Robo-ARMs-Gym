"""Browser classroom so a Windows PC on the LAN can teach the twin.

Opens at https://robo-twin.fimpo.net — no MuJoCo window on the laptop.
"""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from twin.teach import (
    KEY_ENTER,
    KEY_G,
    KEY_K,
    KEY_O,
    KEY_R,
    KEY_SPACE,
    TeachSession,
    mocap_xyz,
    nudge_mocap,
    preview_path,
    set_mocap_xyz,
)

PIXEL_TO_M = 0.0011
WEB_NUDGE_M = 0.02
HOLD_M = 0.007
_MOVE = {
    "left": (-WEB_NUDGE_M, 0.0, 0.0),
    "right": (WEB_NUDGE_M, 0.0, 0.0),
    "forward": (0.0, WEB_NUDGE_M, 0.0),
    "back": (0.0, -WEB_NUDGE_M, 0.0),
    "up": (0.0, 0.0, WEB_NUDGE_M),
    "down": (0.0, 0.0, -WEB_NUDGE_M),
}
# Keep the handle over the table so a few Left clicks cannot throw it off camera.
_MOCAP_LO = (0.12, -0.12, 0.028)
_MOCAP_HI = (0.34, 0.12, 0.22)
_ZOOM_MIN = 0.35
_ZOOM_MAX = 2.4
_ZOOM_STEP = 0.15
_zoom = 1.0
_cam_home: list[float] | None = None

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Twin classroom</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; max-width: 1120px; margin: 0 auto; padding: 1.25rem; line-height: 1.45; }
    h1 { margin: 0 0 0.35rem; font-size: 1.5rem; }
    .lede, .meta { color: #666; }
    .stage { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 1rem; align-items: start; }
    @media (max-width: 860px) { .stage { grid-template-columns: 1fr; } }
    .room .row { margin: 0.5rem 0; }
    canvas.view {
      width: 100%; max-width: none; height: auto; border-radius: 12px;
      border: 1px solid #ccc; cursor: grab; touch-action: none; background: #111;
      display: block;
    }
    canvas.view:active { cursor: grabbing; }
    canvas.view:focus { outline: 2px solid #2e7d32; outline-offset: 2px; }
    aside.monitor {
      border: 1px solid #ccc; border-radius: 12px; padding: 0.75rem 0.85rem;
      font-size: 0.82rem; font-variant-numeric: tabular-nums;
      max-height: calc(100vh - 6rem); overflow: auto; position: sticky; top: 0.75rem;
    }
    aside.monitor h2 { margin: 0 0 0.4rem; font-size: 0.95rem; }
    aside.monitor h3 { margin: 0.7rem 0 0.25rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: #666; }
    aside.monitor table { width: 100%; border-collapse: collapse; }
    aside.monitor th, aside.monitor td { text-align: left; padding: 0.12rem 0.2rem; }
    aside.monitor td.n { text-align: right; font-family: ui-monospace, monospace; }
    aside.monitor ol { margin: 0.2rem 0 0; padding-left: 1.15rem; }
    aside.monitor li { margin: 0.25rem 0; }
    aside.monitor code { font-size: 0.75rem; }
    aside.monitor.rec { border-color: #c62828; }
    .row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.75rem 0; }
    button {
      font: inherit; padding: 0.45rem 0.8rem; border-radius: 8px;
      border: 1px solid #888; background: transparent; cursor: pointer;
    }
    button.primary { border-color: #2e7d32; }
    #status.rec { color: #c62828; font-weight: 600; }
    pre.keys {
      margin: 0.35rem 0 0; padding: 0.55rem 0.7rem; border-radius: 8px;
      border: 1px solid #ccc; font-size: 0.8rem; line-height: 1.4; overflow: auto;
    }
    a { color: inherit; }
  </style>
</head>
<body>
  <h1>Twin classroom</h1>
  <p class="lede">You are in the browser. This page is the 3D room.
  Drag anywhere on the picture to move the RGB sticks — you cannot grab the can
  with the mouse. The claw follows. This is practice, not a license.</p>
  <p class="meta">Skill: <strong>__SKILL__</strong> ·
    <a href="https://robots.fimpo.net/">Robots hub</a> ·
    <a href="https://robo-mission.fimpo.net/floor">Floor board</a></p>
  <div class="stage">
    <div class="room">
      <canvas class="view" id="view" width="640" height="360" aria-label="Digital twin"></canvas>
      <p id="status" class="meta">connecting…</p>
      <div class="row">
        <button type="button" data-cmd="zoom_in">Zoom in</button>
        <button type="button" data-cmd="zoom_out">Zoom out</button>
        <button type="button" data-cmd="zoom_reset">Reset view</button>
      </div>
      <div class="row">
        <button type="button" data-cmd="left">Left</button>
        <button type="button" data-cmd="right">Right</button>
        <button type="button" data-cmd="forward">Forward</button>
        <button type="button" data-cmd="back">Back</button>
        <button type="button" data-cmd="up">Up</button>
        <button type="button" data-cmd="down">Down</button>
      </div>
      <div class="row">
        <button type="button" data-cmd="open">Open claw</button>
        <button type="button" data-cmd="close">Close claw</button>
        <button type="button" data-cmd="record">Record on/off</button>
        <button type="button" data-cmd="waypoint">Save pose</button>
        <button type="button" class="primary" data-cmd="save">Write demo</button>
        <button type="button" data-cmd="reset">Reset room</button>
      </div>
      <pre class="keys">Click the picture, then:
Hold   W A S D  or arrows     move the marker
       Q / E  or  [ / ]       down / up
Tap    K                      start or stop RECORD
       G / O                  close / open claw
       Enter                  write the demo
       R                      reset room
       + / − / 0              zoom</pre>
    </div>
    <aside class="monitor" id="monitor">
      <h2>Live positions</h2>
      <p class="meta" id="mon-rec">idle</p>
      <h3>Joints (deg)</h3>
      <table id="mon-joints"><tbody></tbody></table>
      <h3>World (m)</h3>
      <table>
        <tbody>
          <tr><td>marker</td><td class="n" id="mon-mocap">—</td></tr>
          <tr><td>claw</td><td class="n" id="mon-ee">—</td></tr>
          <tr><td>can</td><td class="n" id="mon-cube">—</td></tr>
          <tr><td>goal pad</td><td class="n" id="mon-goal">—</td></tr>
          <tr><td>fence top</td><td class="n" id="mon-wall">—</td></tr>
        </tbody>
      </table>
      <p class="meta" id="mon-demo"></p>
      <h3>Train</h3>
      <ol>
        <li><strong>K</strong> starts record. Hold WASD to lift. <strong>K</strong> stops. <strong>Enter</strong> writes the demo.</li>
        <li>Reset room and do it again from a new start. A few good lifts beat one long recording.</li>
        <li>On the Linux box:
          <code>.venv/bin/python -m twin train --skill barrier --steps 50000</code></li>
        <li>Exam: <code>.venv/bin/python -m twin eval --skill barrier --policy ppo --model &lt;zip&gt; --episodes 50 --emit-eval</code>. Promote only at ≥ 80%.</li>
      </ol>
      <h3>If you move the object (e.g. up)</h3>
      <ol>
        <li>The old demos no longer match. Reset, then record new lifts that actually go to the new height.</li>
        <li>Train further on the last zip:
          <code>.venv/bin/python -m twin train --skill barrier --resume &lt;zip&gt; --steps 30000</code></li>
        <li>Run the 50-try exam again. A higher can is a new job until the exam passes.</li>
      </ol>
    </aside>
  </div>
<script>
const view = document.getElementById('view');
const ctx = view.getContext('2d');
const status = document.getElementById('status');
let last = null;
let dragging = false;
function xyz(a) {
  if (!a || !a.length) return '—';
  return a.map(x => Number(x).toFixed(3)).join('  ');
}
function paintStatus(j) {
  if (!j || !j.mocap) { status.textContent = 'not connected'; return; }
  status.classList.toggle('rec', !!j.recording);
  status.textContent = (j.recording ? 'RECORDING · ' : '') +
    'claw ' + j.claw + ' · zoom ' + Number(j.zoom || 1).toFixed(2) +
    'x · marker ' + xyz(j.mocap);
  const mon = document.getElementById('monitor');
  mon.classList.toggle('rec', !!j.recording);
  document.getElementById('mon-rec').textContent =
    (j.recording ? 'RECORDING · ' : '') +
    (j.frames || 0) + ' frames · ' + (j.waypoints || 0) + ' poses · claw ' + j.claw;
  const tb = document.querySelector('#mon-joints tbody');
  tb.innerHTML = (j.joints || []).map(row =>
    '<tr><td>' + row.name.split('_').join(' ') + '</td><td class="n">' +
    Number(row.deg).toFixed(1) + '</td></tr>'
  ).join('');
  document.getElementById('mon-mocap').textContent = xyz(j.mocap);
  document.getElementById('mon-ee').textContent = xyz(j.ee);
  document.getElementById('mon-cube').textContent = xyz(j.cube);
  document.getElementById('mon-goal').textContent = xyz(j.goal);
  document.getElementById('mon-wall').textContent = j.wall_top != null
    ? Number(j.wall_top).toFixed(3) + ' m' : '—';
  document.getElementById('mon-demo').textContent = j.last_demo
    ? 'last demo ' + j.last_demo.split('/').pop() : '';
}
let frameBusy = false;
function refresh() {
  if (frameBusy) return;
  frameBusy = true;
  const im = new Image();
  im.onload = () => { ctx.drawImage(im, 0, 0, view.width, view.height); frameBusy = false; };
  im.onerror = () => { frameBusy = false; };
  im.src = '/frame.jpg?t=' + Date.now();
}
setInterval(refresh, 50);
refresh();
let inflight = false;
let queued = null;
let dragAcc = {px: 0, py: 0};
let dragDirty = false;
async function cmd(name, extra) {
  if (inflight) {
    if (name === 'hold' && queued && queued.name === 'hold') {
      queued.extra = extra || {};
      return;
    }
    if ((name === 'zoom_in' || name === 'zoom_out') && queued && queued.name === name) {
      queued.extra = queued.extra || {};
      queued.extra.n = (queued.extra.n || 1) + 1;
      return;
    }
    queued = {name, extra: extra || {}};
    return;
  }
  inflight = true;
  try {
    const r = await fetch('/cmd', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.assign({cmd: name}, extra || {}))
    });
    if (!r.ok) { status.textContent = 'command failed (' + r.status + ')'; return; }
    paintStatus(await r.json());
  } catch (err) {
    status.textContent = 'not connected';
  } finally {
    inflight = false;
    if (queued) {
      const next = queued;
      queued = null;
      cmd(next.name, next.extra);
    }
  }
}
function flushDrag() {
  if (!dragDirty) return;
  dragDirty = false;
  const extra = dragAcc;
  dragAcc = {px: 0, py: 0};
  cmd('drag', extra);
}
setInterval(flushDrag, 32);
view.tabIndex = 0;
view.focus();
const holdMap = {
  ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'forward', ArrowDown: 'back',
  a: 'left', d: 'right', w: 'forward', s: 'back',
  q: 'down', e: 'up', '[': 'down', ']': 'up'
};
const holdDelta = {
  left: [-1, 0, 0], right: [1, 0, 0], forward: [0, 1, 0],
  back: [0, -1, 0], up: [0, 0, 1], down: [0, 0, -1]
};
const held = new Set();
function flushHold() {
  if (!held.size) return;
  let dx = 0, dy = 0, dz = 0;
  held.forEach(name => {
    const d = holdDelta[name];
    dx += d[0]; dy += d[1]; dz += d[2];
  });
  if (dx || dy || dz) cmd('hold', {dx: dx, dy: dy, dz: dz});
}
setInterval(flushHold, 32);
window.addEventListener('blur', () => held.clear());
document.querySelectorAll('button[data-cmd]').forEach(b => {
  b.onclick = () => cmd(b.dataset.cmd);
});
view.addEventListener('pointerdown', e => {
  view.focus();
  dragging = true;
  last = [e.clientX, e.clientY];
  view.setPointerCapture(e.pointerId);
});
function endDrag() {
  dragging = false;
  last = null;
  flushDrag();
}
view.addEventListener('pointerup', endDrag);
view.addEventListener('pointercancel', endDrag);
view.addEventListener('pointermove', e => {
  if (!dragging || last === null) return;
  dragAcc.px += e.clientX - last[0];
  dragAcc.py += e.clientY - last[1];
  last = [e.clientX, e.clientY];
  dragDirty = true;
});
view.addEventListener('wheel', e => {
  e.preventDefault();
  cmd(e.deltaY < 0 ? 'zoom_in' : 'zoom_out');
}, {passive: false});
window.addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (holdMap[k]) {
    e.preventDefault();
    if (!e.repeat) held.add(holdMap[k]);
    return;
  }
  const tap = {
    k: 'record', g: 'close', o: 'open', r: 'reset',
    ' ': 'waypoint', Enter: 'save',
    '=': 'zoom_in', '+': 'zoom_in', '-': 'zoom_out', '0': 'zoom_reset'
  };
  const name = tap[k];
  if (!name) return;
  e.preventDefault();
  if (e.repeat) return;
  if (document.activeElement && document.activeElement.blur) {
    document.activeElement.blur();
  }
  cmd(name);
}, true);
window.addEventListener('keyup', e => {
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (holdMap[k]) {
    e.preventDefault();
    held.delete(holdMap[k]);
  }
});
async function poll() {
  try { paintStatus(await (await fetch('/status')).json()); }
  catch (err) { status.textContent = 'not connected'; }
}
setInterval(poll, 200);
poll();
</script>
</body>
</html>
"""

_CMDS = {
    "open": KEY_O,
    "close": KEY_G,
    "record": KEY_K,
    "waypoint": KEY_SPACE,
    "save": KEY_ENTER,
    "reset": KEY_R,
}


def _clip_mocap(env) -> None:
    xyz = mocap_xyz(env)
    set_mocap_xyz(
        env,
        [min(_MOCAP_HI[i], max(_MOCAP_LO[i], float(xyz[i]))) for i in range(3)],
    )


def apply_hold(env, dx: float, dy: float, dz: float) -> None:
    nudge_mocap(env, [float(dx) * HOLD_M, float(dy) * HOLD_M, float(dz) * HOLD_M])
    _clip_mocap(env)


def _classroom_cam(env) -> int:
    import mujoco

    return int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "classroom"))


def apply_zoom(env, zoom: float | None = None) -> float:
    """Move the classroom camera along the look-at ray. 1.0 is the XML pose."""
    global _zoom, _cam_home
    if zoom is not None:
        _zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, float(zoom)))
    cid = _classroom_cam(env)
    if cid < 0:
        return _zoom
    if _cam_home is None:
        _cam_home = [float(x) for x in env.model.cam_pos[cid]]
    look = [0.26, 0.0, 0.06]
    bid = int(env.model.cam_targetbodyid[cid])
    if bid >= 0:
        look = [float(x) for x in env.data.xpos[bid]]
        look[2] += 0.05
    env.model.cam_pos[cid] = [
        look[i] + (_cam_home[i] - look[i]) * _zoom for i in range(3)
    ]
    return _zoom


def _bump_zoom(cmd: str, n: float = 1.0) -> float:
    global _zoom
    steps = max(1.0, float(n))
    if cmd == "zoom_in":
        _zoom = max(_ZOOM_MIN, _zoom - _ZOOM_STEP * steps)
    elif cmd == "zoom_out":
        _zoom = min(_ZOOM_MAX, _zoom + _ZOOM_STEP * steps)
    else:
        _zoom = 1.0
    return _zoom


class _Camera:
    """Reuse one EGL renderer; JPEG lives in RAM so /cmd never waits on a snapshot."""

    def __init__(self, env) -> None:
        import mujoco

        self.renderer = mujoco.Renderer(env.model, height=360, width=640)
        self.cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "classroom")
        self.jpeg = b""
        self._jlock = threading.Lock()

    def grab(self, env) -> bytes:
        from PIL import Image

        if self.cam >= 0:
            self.renderer.update_scene(env.data, camera=self.cam)
        else:
            self.renderer.update_scene(env.data)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, format="JPEG", quality=55)
        data = buf.getvalue()
        with self._jlock:
            self.jpeg = data
        return data

    def bytes(self) -> bytes:
        with self._jlock:
            return self.jpeg


_lock = threading.Lock()
_session: TeachSession | None = None
_camera: _Camera | None = None
_last: dict[str, Any] = {}


def session_status(session: TeachSession) -> dict[str, Any]:
    env = session.env
    mocap = [float(x) for x in mocap_xyz(env)] if env.model.nmocap else []
    claw = "open" if session.gripper_cmd > 0.5 else "closed"
    joints: list[dict[str, Any]] = []
    names = getattr(env, "joint_ids", None)
    if names is not None:
        from twin.place_env import JOINT_NAMES

        for i, jid in enumerate(env.joint_ids):
            rad = float(env.data.qpos[env.model.jnt_qposadr[jid]])
            label = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"j{i}"
            joints.append({"name": label, "rad": rad, "deg": rad * (180.0 / 3.141592653589793)})
    out: dict[str, Any] = {
        "skill": session.skill,
        "recording": session.recording,
        "frames": len(session.frames),
        "waypoints": len(session.waypoints),
        "claw": claw,
        "mocap": mocap,
        "ee": [float(x) for x in env.ee_pos()],
        "joints": joints,
        "last_demo": session.last_demo,
        "zoom": _zoom,
    }
    if hasattr(env, "cube_pos"):
        out["cube"] = [float(x) for x in env.cube_pos()]
    if hasattr(env, "goal"):
        out["goal"] = [float(x) for x in env.goal]
    if hasattr(env, "wall_top"):
        out["wall_top"] = float(env.wall_top())
        out["wall"] = [float(x) for x in env.wall_pos()]
    if hasattr(env, "gripper_opening"):
        out["jaw"] = float(env.gripper_opening())
    return out


def _status() -> dict[str, Any]:
    assert _session is not None
    return session_status(_session)


def _physics_loop(stop: threading.Event, device: str) -> None:
    global _camera
    n = 0
    next_tick = time.perf_counter()
    assert _session is not None
    apply_zoom(_session.env)
    cam = _Camera(_session.env)
    cam.grab(_session.env)
    _camera = cam
    while not stop.is_set():
        with _lock:
            apply_zoom(_session.env)
            _session.tick()
            n += 1
            if n == 1 or n % 2 == 0:
                cam.grab(_session.env)
        if n == 1 or n % 90 == 0:
            try:
                path = preview_path(_session.skill)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(cam.bytes())
                from fleet.registry import heartbeat

                env = _session.env
                heartbeat(
                    device,
                    skill=_session.skill,
                    status="running",
                    kind="digital-twin",
                    extra={"ee_pos": [float(x) for x in env.ee_pos()], "backend": "mujoco-web"},
                )
            except Exception:
                pass
        next_tick += 1.0 / 30.0
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.perf_counter()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = PAGE.replace("__SKILL__", _session.skill if _session else "barrier")
            self._send(200, html.encode(), "text/html; charset=utf-8")
            return
        if path == "/status":
            with _lock:
                body = json.dumps(_status()).encode()
            self._send(200, body, "application/json")
            return
        if path == "/frame.jpg":
            body = _camera.bytes() if _camera is not None else b""
            if not body:
                self._send(404, b"no frame", "text/plain")
                return
            self._send(200, body, "image/jpeg")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            payload = {}
        cmd = str(payload.get("cmd") or "")
        assert _session is not None
        with _lock:
            if cmd == "drag":
                px = float(payload.get("px") or 0)
                py = float(payload.get("py") or 0)
                nudge_mocap(
                    _session.env,
                    [px * PIXEL_TO_M, -py * PIXEL_TO_M, 0.0],
                )
                _clip_mocap(_session.env)
            elif cmd == "hold":
                apply_hold(
                    _session.env,
                    float(payload.get("dx") or 0),
                    float(payload.get("dy") or 0),
                    float(payload.get("dz") or 0),
                )
            elif cmd in _MOVE:
                nudge_mocap(_session.env, list(_MOVE[cmd]))
                _clip_mocap(_session.env)
            elif cmd in ("zoom_in", "zoom_out", "zoom_reset"):
                _bump_zoom(cmd, float(payload.get("n") or 1))
                apply_zoom(_session.env)
            elif cmd in _CMDS:
                _session.on_key(_CMDS[cmd])
            body = json.dumps(_status()).encode()
        self._send(200, body, "application/json")


def _env_for_skill(skill: str, seed: int):
    if skill == "barrier":
        from twin.barrier_env import BarrierTwin

        return BarrierTwin(seed=seed)
    if skill == "place":
        from twin.place_env import PlaceTwin

        return PlaceTwin(seed=seed)
    from twin.reach_env import ReachTwin

    return ReachTwin(seed=seed)


def serve(
    *,
    skill: str = "barrier",
    host: str = "0.0.0.0",
    port: int = 8092,
    device: str = "so101-twin-web",
    seed: int = 0,
) -> None:
    global _session, _camera, _zoom, _cam_home
    _zoom = 1.0
    _cam_home = None
    env = _env_for_skill(skill, seed)
    _session = TeachSession(env, skill)
    _camera = None
    stop = threading.Event()
    threading.Thread(target=_physics_loop, args=(stop, device), daemon=True).start()
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if _camera is not None and _camera.bytes():
            break
        time.sleep(0.05)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        json.dumps(
            {
                "classroom": "https://robo-twin.fimpo.net/",
                "bind": f"http://{host}:{port}/",
                "skill": skill,
                "hint": "Open the classroom hostname, or the bind address on this network",
            }
        ),
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        httpd.server_close()
