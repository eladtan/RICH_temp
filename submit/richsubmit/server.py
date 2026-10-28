"""The web GUI: a localhost HTTP server for gui/index.html and a small JSON API.

Login nodes are shared, so every request must carry the random token printed at
start-up (the ?token= of the printed URL; the page sends it back as X-Token).
"""

from __future__ import annotations

import json
import re
import secrets
import socket
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from . import jobs, slurm
from .plan import BUILD_MODES, SUBMIT_DIR, Context, RunSpec, make_plan, submit

GUI_PAGE = SUBMIT_DIR / "gui" / "index.html"
SPEC_NAME = re.compile(r"^[\w.-]{1,80}$")
OUTPUT_TAIL = 200_000  # characters of SLURM output shown per file


class Api:
    def __init__(self, context: Context):
        self.context = context
        self.specs_dir = SUBMIT_DIR / "specs"

    def init(self, query) -> Dict:
        site = self.context.site()
        return {
            "repo": str(self.context.repo),
            "cluster": self.context.cluster(),
            "site": site["name"],
            "default_account": site.get("account", ""),
            "accounts": slurm.accounts(),
            "build": {"configs": self.context.build["configs"],
                      "default_config": self.context.build["default_config"],
                      "modes": list(BUILD_MODES)},
            "catalog": self.context.catalog.to_dict(),
        }

    def partitions(self, query) -> Dict:
        refresh = query.get("refresh", ["0"])[0] == "1"
        rules = self.context.site().get("partition_rules", {})
        partitions = []
        for p in self.context.partitions(refresh):
            data = p.to_dict()
            data["max_nodes"] = rules.get(p.name, {}).get("max_nodes")
            data["rule"] = rules.get(p.name, {}).get("reason", "")
            partitions.append(data)
        return {"partitions": partitions}

    def benchmarks(self, query) -> Dict:
        return {"benchmarks": [b.to_dict() for b in self.context.benchmarks(with_options=False)]}

    def benchmark(self, query) -> Dict:
        found = self.context.find_benchmark(query.get("name", [""])[0])
        if not found:
            return {"benchmark": None}
        account = query.get("account", [""])[0]
        data = found.to_dict()
        for variable in data["environment"]:
            variable["example"] = self.context.env_example(variable["name"], found.name, account)
        return {"benchmark": data}

    def plan(self, body) -> Dict:
        return make_plan(self.context, RunSpec.from_dict(body.get("spec"))).to_dict()

    def submit(self, body) -> Dict:
        try:
            result = submit(self.context, RunSpec.from_dict(body.get("spec")), bool(body.get("dry_run")))
        except (ValueError, slurm.SlurmError) as error:
            return {"ok": False, "error": str(error)}
        return dict(result.to_dict(), ok=True)

    def specs(self, query) -> Dict:
        if not self.specs_dir.is_dir():
            return {"specs": []}
        return {"specs": sorted(path.stem for path in self.specs_dir.glob("*.json"))}

    def spec(self, query) -> Dict:
        name = query.get("name", [""])[0]
        if not SPEC_NAME.match(name):
            raise ValueError("Invalid spec name")
        return {"spec": json.loads((self.specs_dir / f"{name}.json").read_text())}

    def save_spec(self, body) -> Dict:
        name = str(body.get("name", ""))
        if not SPEC_NAME.match(name):
            raise ValueError("A spec name is letters, digits, '.', '-' and '_'")
        self.specs_dir.mkdir(exist_ok=True)
        path = self.specs_dir / f"{name}.json"
        path.write_text(json.dumps(RunSpec.from_dict(body.get("spec")).to_dict(), indent=2) + "\n")
        return {"path": str(path)}

    def jobs(self, query) -> Dict:
        return {"jobs": jobs.recent(self.context.jobs_dir)}

    def job_output(self, query) -> Dict:
        """The SLURM output of one job directory (only directories under jobs_dir)."""
        directory = (self.context.jobs_dir / query.get("name", [""])[0]).resolve()
        if directory.parent != self.context.jobs_dir.resolve() or not directory.is_dir():
            raise ValueError("No such job directory")
        files = []
        for path in sorted(directory.glob("*.out")):
            text = path.read_text(errors="replace")
            files.append({"name": path.name, "text": text[-OUTPUT_TAIL:], "truncated": len(text) > OUTPUT_TAIL})
        return {"dir": str(directory), "files": files}

    GET = {"/api/init": "init", "/api/partitions": "partitions", "/api/benchmarks": "benchmarks",
           "/api/benchmark": "benchmark", "/api/specs": "specs", "/api/spec": "spec", "/api/jobs": "jobs",
           "/api/job_output": "job_output"}
    POST = {"/api/plan": "plan", "/api/submit": "submit", "/api/spec": "save_spec"}


def make_handler(api: Api, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rich-submit"

        def log_message(self, format, *args):  # keep the terminal for the URL and errors
            pass

        def _authorized(self, query) -> bool:
            supplied = self.headers.get("X-Token") or query.get("token", [""])[0]
            return secrets.compare_digest(supplied, token)

        def _send(self, status, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, data):
            self._send(status, json.dumps(data).encode(), "application/json")

        def _dispatch(self, table, argument_from_request):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if not self._authorized(query):
                self._send(HTTPStatus.FORBIDDEN,
                           b"Missing or wrong token: open the full URL that rich-submit printed.", "text/plain")
                return
            if url.path in ("/", "/index.html") and table is Api.GET:
                self._send(HTTPStatus.OK, GUI_PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            name = table.get(url.path)
            if not name:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"No endpoint {url.path}"})
                return
            try:
                self._json(HTTPStatus.OK, getattr(api, name)(argument_from_request(query)))
            except (ValueError, KeyError, FileNotFoundError, slurm.SlurmError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # report, keep serving
                traceback.print_exc()
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})

        def do_GET(self):
            self._dispatch(Api.GET, lambda query: query)

        def do_POST(self):
            def body(_):
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"{}")
            self._dispatch(Api.POST, body)

    return Handler


DEFAULT_PORT = 8765  # tried first, so an SSH tunnel command can stay the same between runs


def _listen(handler, port: Optional[int]) -> ThreadingHTTPServer:
    """A server on 127.0.0.1: the given port, or else DEFAULT_PORT or the next free one after it."""
    if port is not None and port != 0:
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), handler)
        except OSError as error:
            raise SystemExit(f"rich-submit: cannot listen on port {port}: {error}. Try another --port.")
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + 20):  # a login node is shared; others may hold one
        try:
            return ThreadingHTTPServer(("127.0.0.1", candidate), handler)
        except OSError:
            continue
    return ThreadingHTTPServer(("127.0.0.1", 0), handler)


def serve(context: Context, port: Optional[int] = None) -> None:
    token = secrets.token_urlsafe(18)
    server = _listen(make_handler(Api(context), token), port)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/?token={token}"
    host = socket.getfqdn()
    login = host_login(host)
    print(f"""
  RICH Submit is running on {host}, port {port}.
  It only listens on this login node, so your browser reaches it through SSH.

  1. Make the tunnel, on YOUR computer, in a new terminal (it asks for your
     PIN+passcode like any login, then stays silent; leave it open):

         ssh -N -L {port}:127.0.0.1:{port} {login}

     It must go to {host.split(".")[0]} exactly: frontier.olcf.ornl.gov may land on another node.
     "Address already in use" means port {port} is busy on your computer (often a VS Code
     forward or an old tunnel): change only the first number, e.g. -L 9876:127.0.0.1:{port},
     and open the link below with 9876 instead of {port}.

     No second login: in the SSH session you are in now, press Enter, type ~C,
     then -L {port}:127.0.0.1:{port} and Enter. (OpenSSH 9.2+ allows ~C only when
     you connected with -o EnableEscapeCommandline=yes.)

     In VS Code connected to this node: Ctrl+click the link below instead.

  2. Open this link in your browser:

         {url}

  Press Ctrl+C here to stop.
""", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        server.server_close()


def host_login(host: str) -> str:
    """The ssh target for this login node: its name, with the user's login."""
    import getpass
    return f"{getpass.getuser()}@{host}"
