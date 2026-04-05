# src/coherent/imagen.py
"""
Authenticated client for the Coherent Imagen service.

Subclasses CoherentAPI so credentials, token refresh, and HTTP plumbing
are inherited — no separate client object needed.

Usage:

    from coherent.imagen import ImagenClient

    img = ImagenClient(username="alice", password="hunter2")

    # Inspect what's available
    img.list_models()
    img.list_loras(model="dreamshaper_xl")

    # Generate — blocks until done, returns a Picture you can use immediately
    pic = img.txt2img("dreamshaper_xl", "a cat on a throne", steps=25, seed=42)
    pic.save("./cat.png")

    # Stream progress while generating
    for event in img.txt2img_stream("dreamshaper_xl", "a cat on a throne"):
        if event["type"] == "progress":
            print(f"  {event['step']}/{event['total']}")
        elif event["type"] == "final":
            event["picture"].save("./cat.png")

    # Your previously generated images
    img.list_images()                          # ["abc123.png", ...]
    pic = img.fetch_image("abc123.png")        # -> Picture
    pic.save("./abc123.png")

    # Job management
    img.list_jobs()
    img.cancel_job("some-uuid")
    for event in img.job_updates("some-uuid"): # re-attach to running job
        ...
"""
from __future__ import annotations

import json
from os import environ as ENV
from pathlib import Path
from typing import Iterator, Optional

from trivialai.image import Picture

from .client import CoherentAPI

_DEFAULT_IMAGEN_URL = ENV.get(
    "COHERENT_IMAGEN_URL",
    "https://imagen.coherentvolition.com",
)


class ImagenClient(CoherentAPI):
    """
    Authenticated client for the Coherent Imagen API.

    Inherits all credential / token-refresh behaviour from CoherentAPI.
    All image-returning methods return Picture instances so you can
    immediately call .save(), .bytes(), .pil_image(), etc.

    Parameters
    ----------
    imagen_url : str, optional
        Base URL of the Imagen service. Falls back to COHERENT_IMAGEN_URL
        env var, then the default production URL.
    username, password, auth_server, persist, timeout
        Passed straight through to CoherentAPI.
    """

    def __init__(
        self,
        *,
        imagen_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_server: Optional[str] = None,
        persist: Optional[bool] = None,
        timeout: float = 15.0,
    ):
        super().__init__(
            username=username,
            password=password,
            auth_server=auth_server,
            persist=persist,
            timeout=timeout,
        )
        self._imagen_url = (imagen_url or _DEFAULT_IMAGEN_URL).rstrip("/")

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    def txt2img(
        self,
        model: str,
        prompt: str,
        *,
        negative_prompt: str = "",
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
        clip_skip: Optional[int] = None,
        loras: Optional[list] = None,
    ) -> Picture:
        """
        Generate an image and return it as a Picture.

        Blocks until generation is complete. Progress events are consumed
        internally. Use txt2img_stream() if you want to observe them.

        Example:
            pic = img.txt2img("dreamshaper_xl", "a cat on a throne", seed=42)
            pic.save("./cat.png")
            pic.pil_image().show()
        """
        for event in self.txt2img_stream(
            model,
            prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            sampler=sampler,
            clip_skip=clip_skip,
            loras=loras,
        ):
            if event["type"] == "final":
                return event["picture"]
            if event["type"] == "error":
                raise RuntimeError(event.get("message", "generation error"))
            if event["type"] == "cancelled":
                raise RuntimeError("job was cancelled")

        raise RuntimeError("stream ended without a final event")

    def txt2img_stream(
        self,
        model: str,
        prompt: str,
        *,
        negative_prompt: str = "",
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
        clip_skip: Optional[int] = None,
        loras: Optional[list] = None,
    ) -> Iterator[dict]:
        """
        Enqueue a txt2img job and yield progress events as they arrive.

        All events pass through unchanged. The "final" event is augmented
        with a "picture" key containing a Picture built from the downloaded
        image bytes, so you can do:

            for event in img.txt2img_stream("dreamshaper", "a cat"):
                if event["type"] == "progress":
                    print(event["step"], "/", event["total"])
                elif event["type"] == "final":
                    event["picture"].save("./cat.png")
        """
        body = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }
        if width is not None:
            body["width"] = width
        if height is not None:
            body["height"] = height
        if seed is not None:
            body["seed"] = seed
        if sampler is not None:
            body["sampler"] = sampler
        if clip_skip is not None:
            body["clip_skip"] = clip_skip
        if loras is not None:
            body["loras"] = loras

        for event in self._ndjson_stream("POST", "/api/txt2img", body):
            if event.get("type") == "final" and "image" in event:
                event["picture"] = self._fetch_picture(event["image"])
            yield event

    def job_updates(self, job_id: str) -> Iterator[dict]:
        """
        Re-attach to an existing job's update stream by job ID.
        Replays all past events then streams new ones live.
        The "final" event will have a "picture" key, same as txt2img_stream.
        """
        for event in self._ndjson_stream("PUT", "/api/txt2img", {"job_id": job_id}):
            if event.get("type") == "final" and "image" in event:
                event["picture"] = self._fetch_picture(event["image"])
            yield event

    # -----------------------------------------------------------------------
    # Job management
    # -----------------------------------------------------------------------

    def list_jobs(self) -> list[dict]:
        """Return the current user's active/queued jobs."""
        return self.get(self._imagen_url, "/api/txt2img")["jobs"]

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a queued or in-progress job."""
        return self.delete(self._imagen_url, "/api/txt2img", body={"job_id": job_id})

    # -----------------------------------------------------------------------
    # Images
    # -----------------------------------------------------------------------

    def list_images(self) -> list[str]:
        """Return filenames of all images generated by the current user."""
        return self.get(self._imagen_url, "/api/images")["images"]

    def fetch_image(self, name: str) -> Picture:
        """
        Download a generated image by filename and return it as a Picture.

        Example:
            pic = img.fetch_image("abc123.png")
            pic.save("./abc123.png")
            pic.pil_image().show()
        """
        return self._fetch_picture(name)

    # -----------------------------------------------------------------------
    # Models / LoRAs
    # -----------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """
        Return all models available on the server, each annotated with
        the LoRAs that are architecturally compatible with it.

        Shape:
            [{"name": str, "filename": str, "arch": str,
              "loras": [{"name": str, "filename": str,
                         "model": str, "flavour": str|None}]},
             ...]
        """
        return self.get(self._imagen_url, "/api/models")["models"]

    def get_model(self, name: str) -> Optional[dict]:
        """Look up a model by name (case-insensitive). Returns None if missing."""
        needle = name.lower()
        for model in self.list_models():
            if model["name"].lower() == needle:
                return model
        return None

    def list_loras(self, model: Optional[str] = None) -> list[dict]:
        """
        Return LoRAs available on the server.

        Parameters
        ----------
        model : str, optional
            If given, return only LoRAs compatible with this model name.
        """
        models = self.list_models()
        if model is not None:
            needle = model.lower()
            models = [m for m in models if m["name"].lower() == needle]
        seen = {}
        for m in models:
            for lora in m.get("loras", []):
                seen[lora["name"]] = lora
        return list(seen.values())

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _fetch_picture(self, name: str) -> Picture:
        """Download an image by filename and wrap it in a Picture."""
        self._ensure_valid_jwt()
        resp = self._http.get(
            f"{self._imagen_url}/api/image/{name}",
            headers={"Authorization": f"Bearer {self.jwt}"},
        )
        resp.raise_for_status()
        return Picture.from_bytes(
            resp.content,
            media_type=resp.headers.get("content-type"),
        )

    def _ndjson_stream(self, method: str, path: str, body: dict) -> Iterator[dict]:
        """
        Open a streaming ndjson response and yield parsed dicts.
        Handles auth and a single transparent re-auth on 401.

        Uses a short connect timeout but no read timeout — the server may
        hold the connection open for several minutes while loading a model.
        """
        import httpx

        self._ensure_valid_jwt()
        url = f"{self._imagen_url}/{path.lstrip('/')}"

        stream_timeout = httpx.Timeout(
            connect=10.0,  # fail fast if the server isn't reachable
            read=None,  # wait indefinitely for model load + generation
            write=10.0,
            pool=10.0,
        )

        def _headers():
            return {
                "Authorization": f"Bearer {self.jwt}",
                "Accept": "application/x-ndjson",
                "Content-Type": "application/json",
            }

        with self._http.stream(
            method,
            url,
            headers=_headers(),
            content=json.dumps(body),
            timeout=stream_timeout,
        ) as resp:
            if resp.status_code == 401:
                self._refresh()
                resp.raise_for_status()
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
