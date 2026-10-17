import urllib.request
import json
from pathlib import Path

def post_multipart(url, fields, files):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))
    for field_name, (filename, data) in files:
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8"))
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

samples_dir = Path(__file__).resolve().parent.parent / "frontend" / "assets" / "samples"
with open(samples_dir / "sample_01_t1.jpg", "rb") as f1, open(samples_dir / "sample_01_b1.jpg", "rb") as f2:
    t1_bytes = f1.read()
    b1_bytes = f2.read()

# 1. Front-View
front_resp = post_multipart("http://127.0.0.1:5000/api/predict/front-view", {}, [("images", ("sample_01_t1.jpg", t1_bytes))])
print("Front-View API:", front_resp["predictions"][0]["predicted_class"], front_resp["predictions"][0]["confidence"], "%")

# 2. Dual-View
dual_resp = post_multipart("http://127.0.0.1:5000/api/predict/dual-view", {}, [
    ("front_image", ("sample_01_t1.jpg", t1_bytes)),
    ("back_image", ("sample_01_b1.jpg", b1_bytes))
])
print("Dual-View API:", dual_resp["predicted_class"], dual_resp["confidence"], "% | Mode:", dual_resp["inference_mode"])

# 3. Batch API
batch_resp = post_multipart("http://127.0.0.1:5000/api/predict/batch", {"mode": "auto"}, [
    ("files", ("sample_01_t1.jpg", t1_bytes)),
    ("files", ("sample_01_b1.jpg", b1_bytes))
])
print("Batch API Samples:", batch_resp["summary"]["total_samples"], "| Class Dist:", batch_resp["summary"]["class_distribution"])
