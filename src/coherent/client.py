import json
import urllib.parse
from datetime import datetime, timezone
from os import environ as ENV
from pathlib import Path

import httpx
import jwt
from dotenv import load_dotenv

load_dotenv()

AUTH_SERVER = ENV.get("AUTH_SERVER", "https://auth.inaimathi.com")


class CoherentAPI:
    def __init__(self, jwt=None, refresh_token=None, timeout: float = 15.0):
        """Initialize the CoherentAPI client."""
        self.jwt = jwt
        self.refresh_token = refresh_token
        self.config_dir = Path.home() / ".coherent"

        # Reusable HTTPX client
        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
            },
        )

        # Ensure the config directory exists
        self.config_dir.mkdir(exist_ok=True)

        if jwt is None:
            self._load_tokens()

    def close(self):
        try:
            self._http.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def auth(self, username, password):
        """
        Authenticate with the API using username and password.
        Stores both JWT and refresh_token in instance variables and on disk.
        """
        url = f"{AUTH_SERVER.rstrip('/')}/api/password/authenticate"
        resp = self._http.post(url, json={"password": password, "username": username})
        resp.raise_for_status()
        response = resp.json()

        # Store tokens in instance variables
        self.jwt = response["jwt"]
        self.refresh_token = response["refresh_token"]

        # Store tokens on disk
        self._save_tokens()

        return response

    def _save_tokens(self):
        """Save authentication tokens to disk."""
        token_file = self.config_dir / "tokens.json"
        with open(token_file, "w") as f:
            json.dump({"jwt": self.jwt, "refresh_token": self.refresh_token}, f)

    def _load_tokens(self):
        """Load authentication tokens from disk if available."""
        token_file = self.config_dir / "tokens.json"
        if token_file.exists():
            try:
                with open(token_file, "r") as f:
                    tokens = json.load(f)
                    self.jwt = tokens.get("jwt")
                    self.refresh_token = tokens.get("refresh_token")
            except (json.JSONDecodeError, KeyError):
                # Handle corrupted token file
                self.jwt = None
                self.refresh_token = None

    def request(self, method, url, path, data=None, public=False):
        """
        Make an HTTP request to the specified URL and path.

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.)
            url (str): Base URL for the request
            path (str): Path to append to the URL
            data (dict, optional): For GET/HEAD/DELETE: query params; otherwise: JSON body
            public (bool, optional): If False, include authentication header

        Returns:
            dict or httpx.Response: Parsed JSON if available, else the raw response
        """
        # Construct the full URL
        full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"

        # Set up headers
        headers = {}

        # Handle authentication if not a public endpoint
        if not public:
            if not self.jwt:
                raise Exception(
                    "Authentication required but no JWT is stored. Call auth() first."
                )

            # Refresh JWT if expired or malformed
            try:
                decoded = jwt.decode(self.jwt, options={"verify_signature": False})
                exp_timestamp = decoded.get("exp")
                if exp_timestamp and datetime.fromtimestamp(
                    exp_timestamp, tz=timezone.utc
                ) <= datetime.now(timezone.utc):
                    self._refresh_token()
            except Exception:
                self._refresh_token()

            headers["Authorization"] = f"Bearer {self.jwt}"

        # Normalize method to uppercase
        method = method.upper()

        # Methods that typically don't have request bodies
        no_body_methods = {"GET", "HEAD", "DELETE", "OPTIONS"}

        # Prepare request arguments
        request_kwargs = {"headers": headers}

        if data:
            if method in no_body_methods:
                # Query parameters for no-body methods
                request_kwargs["params"] = data
            else:
                # JSON body for others
                headers["Content-Type"] = "application/json"
                request_kwargs["json"] = data

        # Make the request
        resp = self._http.request(method, full_url, **request_kwargs)
        resp.raise_for_status()

        # Try to return JSON, but fall back to raw Response if not JSON
        try:
            return resp.json()
        except ValueError:
            return resp

    def _refresh_token(self):
        """
        Refresh the JWT using the stored refresh token.
        """
        if not self.refresh_token:
            raise Exception("No refresh token available. Please re-authenticate.")

        url = f"{AUTH_SERVER.rstrip('/')}/api/token"
        resp = self._http.post(url, json={"refresh_token": self.refresh_token})
        if resp.status_code != 200:
            raise Exception(f"Failed to refresh token: {resp.text}")

        token_data = resp.json()
        self.jwt = token_data["jwt"]

        # The response might contain a new refresh token
        if "refresh_token" in token_data:
            self.refresh_token = token_data["refresh_token"]

        self._save_tokens()
