"""Print the tail of a kernel session's log while it runs (or its saved log once it has ended), with an in-memory
access token (credentials.json untouched, token never printed).

    python kaggle/tools/kernel_logs.py <user/kernel-slug> [tail_lines] [listen_seconds]

The live stream is Server-Sent Events; it is read for at most listen_seconds (default 20), then the last lines received are printed.
"""
import json
import sys
import threading

from kagglesdk import KaggleClient
from kagglesdk.kaggle_creds import KaggleCredentials
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

user, slug = sys.argv[1].split("/")
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 40
listen = float(sys.argv[3]) if len(sys.argv) > 3 else 20
lines = []


def read(client):
    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name, req.kernel_slug = user, slug
    resp = client.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
    if hasattr(resp, "iter_lines"):   # line by line: .content would block until a live stream ends
        try:
            for raw in resp.iter_lines():
                lines.append(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw))
        except Exception as e:   # the server sometimes cuts the stream; keep what arrived
            lines.append(f"(stream ended: {e.__class__.__name__})")
    else:
        body = resp.content if hasattr(resp, "content") else resp
        lines.extend((body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)).splitlines())


with KaggleClient() as c0:
    token = KaggleCredentials.load(client=c0).generate_access_token().token
with KaggleClient(api_token=token) as client:
    t = threading.Thread(target=read, args=(client,), daemon=True)
    t.start()
    t.join(listen)
out = []
for ln in lines:   # SSE "data: {...}" lines or a persisted JSON log
    s = ln[5:].strip() if ln.startswith("data:") else ln.strip()
    try:
        d = json.loads(s)
        if isinstance(d, list):
            out += [str(x.get("data", x)) for x in d]
        elif isinstance(d, dict):
            out.append(str(d.get("data", d)))
        else:
            out.append(s)
    except Exception:
        if s:
            out.append(s)
print("\n".join(x.rstrip() for x in out[-tail:]))
