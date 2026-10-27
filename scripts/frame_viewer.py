"""Browser framing view for a figure, served on localhost.

A subsampled cloud is shown with three.js. The page is meant to be opened
through an SSH tunnel. Saving a viewpoint writes a Blender camera block next
to the renders. The final image is still produced by examples/00_custom.py.

    python scripts/frame_viewer.py --config configs/figures/dales_5100_54440_new.yaml
"""
import argparse
import json
import os
import os.path as osp
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from figlib import load_config, load_point_cloud, place_for_figure

# Paper stills are 2100x1190. The viewer keeps that aspect so the crop matches.
ASPECT = 2100 / 1190


def look_at_quaternion(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Blender camera quaternion [w, x, y, z]. The camera looks along local -Z."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    m = np.stack([x_axis, y_axis, z_axis], axis=1)
    trace = np.trace(m)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def world_to_three(xyz):
    """Z-up world (x, y, z) to three.js Y-up (x, z, -y)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    out = np.empty_like(xyz)
    out[..., 0] = xyz[..., 0]
    out[..., 1] = xyz[..., 2]
    out[..., 2] = -xyz[..., 1]
    return out


def three_to_world(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.array([xyz[0], -xyz[2], xyz[1]], dtype=np.float64)


def vertical_fov_deg(fov_x_deg, aspect=ASPECT):
    fov_x = np.deg2rad(fov_x_deg)
    return float(np.rad2deg(2 * np.arctan(np.tan(fov_x / 2) / aspect)))


PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Cadrage</title>
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #eee;
    font-family: sans-serif; }
  #bar { display: flex; gap: 12px; align-items: center; padding: 8px 12px;
    background: #1b1b1b; }
  button, select { font: inherit; padding: 6px 10px; }
  #status { opacity: 0.8; }
  #wrap { height: calc(100% - 48px); display: grid; place-items: center;
    padding: 16px; box-sizing: border-box; }
  #frame { border: 3px solid #fff; line-height: 0;
    width: min(calc(100vw - 48px), calc((100vh - 96px) * 2100 / 1190));
    aspect-ratio: 2100 / 1190; }
  canvas { width: 100%; height: 100%; display: block; }
</style>
</head>
<body>
<div id="bar">
  <label>Couche <select id="layer"></select></label>
  <label>Taille des points <input id="size" type="range" min="0.4" max="4" step="0.1" value="1.6"></label>
  <button id="save" type="button">Enregistrer ce cadrage</button>
  <span id="status">Glisser pour tourner, molette pour zoomer, clic droit pour déplacer. Le cadre blanc est le rendu.</span>
</div>
<div id="wrap"><div id="frame"><canvas id="view"></canvas></div></div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const meta = await (await fetch("/meta.json")).json();
const pos = new Float32Array(await (await fetch("/points.bin")).arrayBuffer());
const colors = {};
for (const name of meta.layers) {
  colors[name] = new Uint8Array(await (await fetch("/colors/" + name + ".bin")).arrayBuffer());
}

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setClearColor(0x111111);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  meta.fov_y_deg, 2100 / 1190, meta.near, meta.far);
camera.position.set(...meta.eye);
const controls = new OrbitControls(camera, canvas);
controls.target.set(...meta.target);
controls.update();

const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
const colorAttr = new THREE.BufferAttribute(colors[meta.layers[0]], 3, true);
geometry.setAttribute("color", colorAttr);
const material = new THREE.PointsMaterial({
  size: 1.6, vertexColors: true, sizeAttenuation: true });
scene.add(new THREE.Points(geometry, material));

const layer = document.getElementById("layer");
for (const name of meta.layers) {
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  layer.appendChild(option);
}
layer.onchange = () => {
  colorAttr.array = colors[layer.value];
  colorAttr.needsUpdate = true;
};
document.getElementById("size").oninput = (event) => {
  material.size = Number(event.target.value);
};

function frame() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  camera.aspect = 2100 / 1190;
  camera.updateProjectionMatrix();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

document.getElementById("save").onclick = async () => {
  const status = document.getElementById("status");
  status.textContent = "Enregistrement…";
  const response = await fetch("/pose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      eye: camera.position.toArray(),
      target: controls.target.toArray(),
      fov_y_deg: camera.fov,
    }),
  });
  const saved = await response.json();
  status.textContent = "Cadrage écrit dans " + saved.path;
};
</script>
</body>
</html>
"""


class Viewer:
    def __init__(self, cloud, layers, eye, target, fov_y_deg, near, far, pose_path):
        positions = np.ascontiguousarray(world_to_three(cloud.pos))
        self.points = positions.astype(np.float32).tobytes()
        self.colors = {
            name: np.ascontiguousarray(cloud.colors[name]).astype(np.uint8).tobytes()
            for name in layers
        }
        self.meta = {
            "layers": list(layers),
            "count": int(len(cloud)),
            "eye": world_to_three(eye).tolist(),
            "target": world_to_three(target).tolist(),
            "fov_y_deg": fov_y_deg,
            "near": near,
            "far": far,
        }
        self.pose_path = pose_path
        self.fov_x_deg = None

    def save_pose(self, payload):
        eye = three_to_world(payload["eye"])
        target = three_to_world(payload["target"])
        quat = look_at_quaternion(eye, target)
        pose = {
            "translation": [round(float(v), 7) for v in eye],
            "quaternion": [round(float(v), 7) for v in quat],
            "target": [round(float(v), 7) for v in target],
            "fov_x_deg": self.fov_x_deg,
        }
        os.makedirs(osp.dirname(self.pose_path), exist_ok=True)
        with open(self.pose_path, "w") as handle:
            json.dump(pose, handle, indent=2)
            handle.write("\n")
        print("Saved viewpoint to", self.pose_path)
        print("camera:")
        print(f"  translation: {pose['translation']}")
        print(f"  quaternion: {pose['quaternion']}")
        return pose


def make_handler(viewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, body, content_type, code=200):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(PAGE, "text/html; charset=utf-8")
            elif self.path == "/meta.json":
                self._send(json.dumps(viewer.meta), "application/json")
            elif self.path == "/points.bin":
                self._send(viewer.points, "application/octet-stream")
            elif self.path.startswith("/colors/") and self.path.endswith(".bin"):
                name = self.path[len("/colors/"):-len(".bin")]
                if name not in viewer.colors:
                    self._send(b"unknown layer", "text/plain", 404)
                    return
                self._send(viewer.colors[name], "application/octet-stream")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/pose":
                self._send(b"not found", "text/plain", 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            pose = viewer.save_pose(payload)
            self._send(json.dumps({"path": viewer.pose_path, "camera": pose}),
                       "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8755)
    parser.add_argument("--subsample", type=int, default=150000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    camera = cfg["camera"]
    cloud = load_point_cloud(
        data["path"],
        palettes=data["palettes"],
        palette_overrides=data["palette_overrides"],
        cache=False,
    )
    place_for_figure(
        cloud, drop_void=data.get("drop_void"), scale=data.get("scale"),
        center=data.get("center", True), grid_size=data.get("grid_size"),
        drop_after_center=data.get("drop_after_center", False))
    cloud.subsample(args.subsample, seed=0)
    preferred = ["rgb", "strength", "segment", "category", "pred", "pred_sonata"]
    default = data.get("default_color")
    if default in cloud.colors:
        preferred = [default] + [name for name in preferred if name != default]
    layers = [name for name in preferred if name in cloud.colors]
    layers += [name for name in cloud.names if name not in layers]
    if not layers:
        raise SystemExit(f"No colorizations in {data['path']}")

    eye = np.asarray(camera["translation"], dtype=np.float64)
    # The stored quaternion looks along the camera's -Z. Aim the orbit target
    # a few hundred metres in front of it, which is where the tile sits.
    quat = np.asarray(camera["quaternion"], dtype=np.float64)
    w, x, y, z = quat
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    forward = rotation @ np.array([0.0, 0.0, -1.0])
    distance = float(np.linalg.norm(eye))
    target = eye + forward * distance

    output_dir = data.get("output_dir") or osp.dirname(data["path"])
    pose_path = osp.join(output_dir, "viewer_pose.json")
    fov_x = float(camera["fov_x_deg"])
    viewer = Viewer(
        cloud, layers, eye, target,
        vertical_fov_deg(fov_x),
        float(camera.get("near", 0.1)),
        float(camera.get("far", 2000)),
        pose_path,
    )
    viewer.fov_x_deg = fov_x

    server = ThreadingHTTPServer((args.host, args.port), make_handler(viewer))
    print(
        f"Frame viewer ready on http://{args.host}:{args.port} "
        f"({viewer.meta['count']} points, layers {layers})")
    print(f"Tunnel from your computer: ssh -L {args.port}:127.0.0.1:{args.port} hecate")
    print("Then open http://127.0.0.1:%d" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
