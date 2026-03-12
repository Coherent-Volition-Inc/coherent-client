# src/coherent/client.py
"""
General-purpose authenticated client for the Coherent network.

Handles JWT acquisition, transparent refresh via cookie jar, and
re-authentication from credentials when the session expires. Works
both as a long-running service account client and as an interactive
REPL client.

Usage — credentials from env (recommended for services and REPL):

    from coherent.client import CoherentAPI
    client = CoherentAPI()   # reads COHERENT_USERNAME / COHERENT_PASSWORD
    result = client.get("https://auth.coherentvolition.com", "/api/me/oauth")

Usage — explicit credentials (e.g. multiple accounts, testing):

    client = CoherentAPI(username="alice@example.com", password="hunter2")

Usage — persistent interactive session (REPL, CLI tools):

    client = CoherentAPI(persist=True)
    # On first run: authenticates and saves tokens to ~/.coherent/tokens.json
    # On subsequent runs: loads tokens from disk, refreshes silently if needed.

Making requests (all methods take a full target URL, not just a path):

    client.get("https://someservice.coherentvolition.com", "/api/items")
    client.post("https://auth.coherentvolition.com", "/api/invites", {"user_group": "x"})
    client.request("DELETE", "https://auth.coherentvolition.com", "/api/token")
"""
import json
from datetime import datetime, timezone
from os import environ as ENV
from pathlib import Path

import httpx
import jwt
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_AUTH_SERVER = ENV.get("COHERENT_AUTH_SERVER", "https://auth.inaimathi.com")
_DEFAULT_USERNAME = ENV.get("COHERENT_USERNAME", "")
_DEFAULT_PASSWORD = ENV.get("COHERENT_PASSWORD", "")
_TOKEN_FILE = Path.home() / ".coherent" / "tokens.json"


class CoherentAPI:
    """
    Authenticated HTTP client for the Coherent network.

    Parameters
    ----------
    username : str, optional
        Account username. Falls back to COHERENT_USERNAME env var.
    password : str, optional
        Account password. Falls back to COHERENT_PASSWORD env var.
    auth_server : str, optional
        Base URL of the auth service. Falls back to COHERENT_AUTH_SERVER
        env var, then "https://auth.coherentvolition.com".
    persist : bool, optional
        If True, save JWT and cookies to ~/.coherent/tokens.json and reload
        them on next instantiation. Useful for interactive/REPL use where
        you don't want to re-authenticate every session.
        Default: False when credentials are provided, True otherwise.
    timeout : float, optional
        HTTP request timeout in seconds. Default: 15.0.
    """

    def __init__(
        self,
        username=None,
        password=None,
        auth_server=None,
        persist=None,
        timeout=15.0,
    ):
        self._username = username or _DEFAULT_USERNAME
        self._password = password or _DEFAULT_PASSWORD
        self._auth_server = (auth_server or _DEFAULT_AUTH_SERVER).rstrip("/")

        # Default persistence: on when no credentials are hardcoded (REPL/CLI
        # use), off when credentials are explicitly provided (service use).
        has_creds = bool(self._username and self._password)
        self._persist = persist if persist is not None else (not has_creds)

        self.jwt = None

        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "CoherentAPI/1.0",
                "Accept": "application/json",
            },
        )

        if self._persist:
            _TOKEN_FILE.parent.mkdir(exist_ok=True)
            self._load_tokens()

        # Authenticate immediately if we have credentials, regardless of
        # whether tokens were loaded from disk — service accounts should be
        # ready to make requests the moment the client is constructed.
        if has_creds and not self.jwt:
            self._authenticate()

    # -----------------------------------------------------------------------
    # Context manager / cleanup
    # -----------------------------------------------------------------------

    def close(self):
        """Close the underlying HTTP client and release connections."""
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    # -----------------------------------------------------------------------
    # HTTP surface
    #
    # All methods take:
    #   url   - full base URL of the target service
    #   path  - path on that service, e.g. "/api/items"
    #   body  - optional payload:
    #             GET/DELETE/HEAD/OPTIONS → sent as query params
    #             POST/PATCH/PUT         → sent as JSON body
    #   public - if True, omit Authorization header (for unauthenticated endpoints)
    # -----------------------------------------------------------------------

    def request(self, method, url, path, body=None, public=False):
        """Make an authenticated HTTP request to any Coherent network service."""
        return self._request(method, url, path, body=body, public=public)

    def get(self, url, path, body=None, public=False):
        return self._request("GET", url, path, body=body, public=public)

    def post(self, url, path, body=None, public=False):
        return self._request("POST", url, path, body=body, public=public)

    def put(self, url, path, body=None, public=False):
        return self._request("PUT", url, path, body=body, public=public)

    def patch(self, url, path, body=None, public=False):
        return self._request("PATCH", url, path, body=body, public=public)

    def delete(self, url, path, body=None, public=False):
        return self._request("DELETE", url, path, body=body, public=public)

    # -----------------------------------------------------------------------
    # Internal request dispatch
    # -----------------------------------------------------------------------

    def _request(self, method, url, path, body=None, public=False):
        full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"
        method = method.upper()
        headers = {}

        if not public:
            self._ensure_valid_jwt()
            headers["Authorization"] = f"Bearer {self.jwt}"

        kwargs = {"headers": headers}
        if body is not None:
            if method in {"GET", "HEAD", "DELETE", "OPTIONS"}:
                kwargs["params"] = body
            else:
                kwargs["json"] = body

        resp = self._http.request(method, full_url, **kwargs)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return resp

    # -----------------------------------------------------------------------
    # Auth lifecycle
    # -----------------------------------------------------------------------

    def _ensure_valid_jwt(self):
        """Guarantee self.jwt is present and not about to expire."""
        if not self.jwt or self._jwt_expiring():
            self._refresh()

    def _authenticate(self):
        """
        Full credential login. Updates self.jwt, populates cookie jar,
        and persists if enabled. Raises if credentials are missing or rejected.
        """
        if not self._username or not self._password:
            raise RuntimeError(
                "No credentials available. Provide username/password or set "
                "COHERENT_USERNAME / COHERENT_PASSWORD."
            )
        resp = self._http.post(
            f"{self._auth_server}/api/password/authenticate",
            data={"username": self._username, "password": self._password},
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "ok" or "jwt" not in body:
            raise RuntimeError(f"Authentication failed: {body}")
        self.jwt = body["jwt"]
        if self._persist:
            self._save_tokens()

    def _refresh(self):
        """
        Attempt to refresh via the refresh_token cookie in the jar.

        On success: updates self.jwt (and persists if enabled).
        On failure:
          - If credentials are available: falls back to full re-auth.
          - Otherwise: raises, since there is no credential-based fallback
            (interactive user must construct a new client with credentials).
        """
        resp = self._http.post(f"{self._auth_server}/api/token")
        if resp.status_code == 200:
            body = resp.json()
            if body.get("status") == "ok" and "jwt" in body:
                self.jwt = body["jwt"]
                if self._persist:
                    self._save_tokens()
                return

        if self._username and self._password:
            self._authenticate()
        else:
            raise RuntimeError(
                "Session expired and no credentials available to re-authenticate. "
                "Construct a new CoherentAPI with username/password."
            )

    def _jwt_expiring(self):
        """
        Return True if the JWT is missing, invalid, or within 60 seconds
        of expiry (to avoid clock-skew races on the server side).
        """
        if not self.jwt:
            return True
        try:
            claims = jwt.decode(self.jwt, options={"verify_signature": False})
            exp = claims.get("exp")
            if not exp:
                return False
            return datetime.now(timezone.utc).timestamp() >= (exp - 60)
        except Exception:
            return True

    # -----------------------------------------------------------------------
    # Token persistence
    # -----------------------------------------------------------------------

    def _save_tokens(self):
        try:
            _TOKEN_FILE.parent.mkdir(exist_ok=True)
            cookies = dict(self._http.cookies)
            with open(_TOKEN_FILE, "w") as f:
                json.dump({"jwt": self.jwt, "cookies": cookies}, f)
        except Exception:
            pass  # persistence failure is non-fatal

    def _load_tokens(self):
        if not _TOKEN_FILE.exists():
            return
        try:
            with open(_TOKEN_FILE) as f:
                data = json.load(f)
            self.jwt = data.get("jwt")
            for name, value in (data.get("cookies") or {}).items():
                self._http.cookies.set(name, value)
        except Exception:
            self.jwt = None
