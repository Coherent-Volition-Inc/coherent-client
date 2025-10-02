import json
import urllib.parse
from datetime import datetime, timezone
from os import environ as ENV
from pathlib import Path

import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

AUTH_SERVER = ENV.get("AUTH_SERVER", "https://auth.inaimathi.com")


class CoherentAPI:
    def __init__(self, jwt=None, refresh_token=None):
        """Initialize the CoherentAPI client."""
        self.jwt = jwt
        self.refresh_token = refresh_token
        self.config_dir = Path.home() / ".coherent"

        # Ensure the config directory exists
        self.config_dir.mkdir(exist_ok=True)

        if jwt is None:
            self._load_tokens()

    def auth(self, username, password):
        """
        Authenticate with the API using username and password.
        Stores both JWT and refresh_token in instance variables and on disk.

        Args:
            username (str): The username for authentication
            password (str): The password for authentication

        Returns:
            dict: The authentication response with JWT and refresh token
        """
        response = requests.post(
            f"{AUTH_SERVER.rstrip('/')}/api/password/authenticate",
            json={"password": password, "username": username},
        ).json()

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
            data (dict, optional): Dictionary of data to include in the request
                                   (as query params for GET/HEAD/DELETE, or JSON body otherwise)
            public (bool, optional): If False, include authentication header. Defaults to False.

        Returns:
            Response: The response from the API

        Raises:
            Exception: If authentication is required but no JWT is stored
        """
        # Construct the full URL
        full_url = f"{url.rstrip('/')}/{path.lstrip('/')}"

        # Set up headers
        headers = {}

        # Handle authentication if not a public endpoint
        if not public:
            # Check if we have a JWT
            if not self.jwt:
                raise Exception(
                    "Authentication required but no JWT is stored. Call auth() first."
                )

            # Check if JWT is expired
            try:
                decoded = jwt.decode(self.jwt, options={"verify_signature": False})
                exp_timestamp = decoded.get("exp")

                if exp_timestamp and datetime.fromtimestamp(
                    exp_timestamp, tz=timezone.utc
                ) <= datetime.now(timezone.utc):
                    self._refresh_token()
            except:
                self._refresh_token()

            # Add the Authorization header
            headers["Authorization"] = f"Bearer {self.jwt}"

        # Normalize method to uppercase
        method = method.upper()

        # Methods that typically don't have request bodies
        no_body_methods = ["GET", "HEAD", "DELETE", "OPTIONS"]

        # Prepare request arguments
        request_kwargs = {"headers": headers}

        if data:
            if method in no_body_methods:
                # For methods that don't typically have bodies, add data as query parameters
                query_string = urllib.parse.urlencode(data)
                full_url = f"{full_url}?{query_string}"
            else:
                # For methods that can have bodies, add data as JSON body
                headers["Content-Type"] = "application/json"
                request_kwargs["json"] = data

        # Make the request
        response = requests.request(method, full_url, **request_kwargs)

        # Raise an exception for HTTP errors
        response.raise_for_status()

        # Try to return JSON, but fall back to regular response if not JSON
        try:
            return response.json()
        except ValueError:
            return response

    def _refresh_token(self):
        """
        Refresh the JWT using the stored refresh token.

        Raises:
            Exception: If no refresh token is stored or if refresh fails
        """
        if not self.refresh_token:
            raise Exception("No refresh token available. Please re-authenticate.")

        response = requests.post(
            f"{AUTH_SERVER}/api/token",
            json={"refresh_token": self.refresh_token},
        )

        # Check for a successful response
        if response.status_code != 200:
            raise Exception(f"Failed to refresh token: {response.text}")

        # Update the stored tokens
        token_data = response.json()
        self.jwt = token_data["jwt"]

        # The response might contain a new refresh token
        if "refresh_token" in token_data:
            self.refresh_token = token_data["refresh_token"]

        # Save the updated tokens
        self._save_tokens()
