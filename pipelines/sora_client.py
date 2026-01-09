"""
Sora API Client for Azure OpenAI.

Usage:
    export SORA_HOST="your-host.azure-api.net"
    export SORA_API_KEY="your-api-key"
    
    from pipelines.sora_client import SoraClient
    
    client = SoraClient()
    client.generate_and_download("A cat playing", "output.mp4")
    client.generate_and_download("Scene comes alive", "output.mp4", input_reference="image.jpg")
"""

import os
import io
import time
import requests
from dataclasses import dataclass
from typing import Optional, Union
from PIL import Image


TIMEOUT = 60
POLL_INTERVAL = 10
MAX_WAIT = 600


@dataclass
class VideoResult:
    video_id: str
    status: str
    progress: int = 0
    error: Optional[str] = None


class SoraClient:
    """Sora video generation client. Reads SORA_HOST and SORA_API_KEY from environment."""
    
    def __init__(self, host: str = None, api_key: str = None):
        self.host = host or os.environ.get("SORA_HOST")
        self.api_key = api_key or os.environ.get("SORA_API_KEY")
        
        if not self.host or not self.api_key:
            raise ValueError("Set SORA_HOST and SORA_API_KEY environment variables")
        
        self.base_url = f"https://{self.host}/openai/v1"
    
    def _prepare_image(self, image: Union[str, Image.Image], size: str) -> tuple:
        """Load and resize image to match video dimensions."""
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        
        img = img.convert("RGB")
        w, h = map(int, size.split("x"))
        img = img.resize((w, h), Image.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return ("image.jpg", buf.getvalue(), "image/jpeg")
    
    def generate(
        self,
        prompt: str,
        input_reference: Union[str, Image.Image] = None,
        seconds: int = 4,
        size: str = "1280x720",
        wait: bool = True,
    ) -> VideoResult:
        """
        Generate a video.
        
        Args:
            prompt: Text description.
            input_reference: Image path or PIL Image for I2V (optional).
            seconds: Duration (4, 8, or 12).
            size: Video size.
            wait: Wait for completion.
        """
        url = f"{self.base_url}/videos"
        data = {"model": "sora-2", "prompt": prompt, "seconds": str(seconds), "size": size}
        
        if input_reference:
            # I2V: multipart form
            files = {"input_reference": self._prepare_image(input_reference, size)}
            resp = self._post_with_retry(url, files=files, data=data)
        else:
            # T2V: JSON
            resp = self._post_with_retry(url, json=data)
        
        result = VideoResult(video_id=resp["id"], status=resp.get("status", "unknown"))
        
        if wait and result.status in ["queued", "in_progress"]:
            return self._poll(result.video_id)
        return result
    
    def _post_with_retry(self, url: str, **kwargs) -> dict:
        """POST with retry on rate limit."""
        headers = {"api-key": self.api_key}
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        
        for attempt in range(4):
            resp = requests.post(url, headers=headers, timeout=TIMEOUT, **kwargs)
            
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            
            if not resp.ok:
                try:
                    msg = resp.json().get("error", {}).get("message", resp.text)
                except:
                    msg = resp.text
                raise RuntimeError(f"API error {resp.status_code}: {msg}")
            
            return resp.json()
        
        raise RuntimeError("Rate limited after retries")
    
    def _poll(self, video_id: str) -> VideoResult:
        """Poll until complete."""
        start = time.time()
        
        while time.time() - start < MAX_WAIT:
            resp = requests.get(
                f"{self.base_url}/videos/{video_id}",
                headers={"api-key": self.api_key},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            
            status = data.get("status", "unknown")
            progress = data.get("progress", 0)
            print(f"  Status: {status}, Progress: {progress}%")
            
            if status == "completed":
                return VideoResult(video_id, status, progress)
            if status in ["failed", "cancelled"]:
                err = data.get("error", {}).get("message", "Failed")
                return VideoResult(video_id, status, progress, err)
            
            time.sleep(POLL_INTERVAL)
        
        return VideoResult(video_id, "timeout", 0, f"Timed out after {MAX_WAIT}s")
    
    def download(self, video_id: str, output_path: str) -> bool:
        """Download completed video."""
        resp = requests.get(
            f"{self.base_url}/videos/{video_id}/content",
            headers={"api-key": self.api_key},
            timeout=120,
        )
        resp.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True
    
    def generate_and_download(
        self,
        prompt: str,
        output_path: str,
        input_reference: Union[str, Image.Image] = None,
        seconds: int = 4,
        size: str = "1280x720",
    ) -> bool:
        """Generate and save video to file."""
        print(f"  Generating...")
        result = self.generate(prompt, input_reference, seconds, size, wait=True)
        
        if result.status != "completed":
            raise RuntimeError(f"Failed: {result.status} - {result.error}")
        
        print(f"  Downloading...")
        self.download(result.video_id, output_path)
        print(f"  Saved: {output_path}")
        return True
