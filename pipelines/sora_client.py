"""
Simple Sora API client for text-to-video generation.
Configured for Azure OpenAI endpoints.
"""

import os
import time
import requests
from requests.exceptions import RequestException, Timeout


# Default Azure configuration
DEFAULT_HOST = "azure-services-fair-openai2-eastus2n6.azure-api.net"
DEFAULT_API_KEY = "a60a6c3d975747ef9866d1827c266976"

# Request timeout in seconds
REQUEST_TIMEOUT = 60


class SoraClient:
    """Simple client for Sora API via Azure OpenAI."""
    
    def __init__(
        self,
        host: str = None,
        api_key: str = None,
    ):
        """
        Initialize the Sora client.
        
        Args:
            host: Azure API host. Defaults to configured endpoint.
            api_key: API key. Defaults to configured key or SORA_API_KEY env var.
        """
        self.host = host or os.environ.get("SORA_HOST", DEFAULT_HOST)
        self.api_key = api_key or os.environ.get("SORA_API_KEY", DEFAULT_API_KEY)
        self.base_url = f"https://{self.host}"
        
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
    
    def generate_video(
        self,
        prompt: str,
        model: str = "sora-2",
        seconds: int = 4,
        size: str = "1280x720",
    ) -> dict:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text description of the video to generate.
            model: Model to use (default: "sora-2").
            seconds: Video duration in seconds (4, 8, or 12).
            size: Video size as "WxH" (e.g., "1280x720", "1920x1080").
            
        Returns:
            Response dict containing video ID and status.
        """
        url = f"{self.base_url}/openai/v1/videos"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "seconds": str(seconds),  # API expects string
            "size": size,
        }
        
        # Retry on rate limits
        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(url, json=payload, headers=self.headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 429:
                wait_time = 30 * (attempt + 1)  # 30, 60, 90, 120, 150 seconds
                print(f"Rate limited (429), waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            if not response.ok:
                print(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()
            return response.json()
        
        raise RuntimeError("Max retries exceeded due to rate limiting")
    
    def get_video_status(self, video_id: str, retries: int = 3) -> dict:
        """
        Check the status of a video generation job with retry logic.
        
        Args:
            video_id: The video ID from generate_video().
            retries: Number of retries on timeout/connection errors.
            
        Returns:
            Response dict with status and progress.
        """
        url = f"{self.base_url}/openai/v1/videos/{video_id}"
        
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except (Timeout, RequestException) as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    print(f"Request failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
        
        raise last_error
    
    def wait_for_video(
        self,
        response: dict,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        verbose: bool = True,
    ) -> str:
        """
        Poll until video generation is complete and return video ID.
        
        Args:
            response: Response dict from generate_video().
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.
            verbose: Print status updates.
            
        Returns:
            Video ID (use download_video to get the content).
        """
        start_time = time.time()
        video_id = response.get("id")
        
        if not video_id:
            raise ValueError("No video ID in response")
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Video generation timed out after {timeout}s")
            
            try:
                status_response = self.get_video_status(video_id)
            except RequestException as e:
                if verbose:
                    print(f"Status check failed: {e}, will retry...")
                time.sleep(poll_interval)
                continue
            
            status = status_response.get("status", "unknown")
            progress = status_response.get("progress", 0)
            
            if verbose:
                print(f"Status: {status}, Progress: {progress}% ({elapsed:.1f}s elapsed)")
            
            if status == "completed":
                if verbose:
                    print(f"Video generation complete! ({elapsed:.1f}s)")
                return video_id
            
            elif status == "failed":
                error = status_response.get("error", "Unknown error")
                raise RuntimeError(f"Video generation failed: {error}")
            
            time.sleep(poll_interval)
    
    def download_video(self, video_id: str, output_path: str, retries: int = 3) -> str:
        """
        Download a video by ID to local file.
        
        Args:
            video_id: Video ID from generate_video/wait_for_video.
            output_path: Local path to save the video.
            retries: Number of retries on timeout/connection errors.
            
        Returns:
            Path to the saved video.
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        url = f"{self.base_url}/openai/v1/videos/{video_id}/content"
        
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return output_path
            except (Timeout, RequestException) as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Download failed, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
        
        raise last_error
    
    def generate_and_download(
        self,
        prompt: str,
        output_path: str,
        model: str = "sora-2",
        seconds: int = 4,
        size: str = "1280x720",
        verbose: bool = True,
    ) -> str:
        """
        Generate a video and download it in one call.
        
        Args:
            prompt: Text description of the video.
            output_path: Local path to save the video.
            model: Model to use.
            seconds: Video duration in seconds (4, 8, or 12).
            size: Video size as "WxH".
            verbose: Print status updates.
            
        Returns:
            Path to the saved video.
        """
        if verbose:
            print(f"Generating video: {prompt[:50]}...")
        
        response = self.generate_video(
            prompt=prompt,
            model=model,
            seconds=seconds,
            size=size,
        )
        
        if verbose:
            print(f"Video queued: {response.get('id')}")
        
        video_id = self.wait_for_video(response, verbose=verbose)
        
        if verbose:
            print(f"Downloading to {output_path}...")
        
        self.download_video(video_id, output_path)
        
        if verbose:
            print(f"Saved: {output_path}")
        
        return output_path
