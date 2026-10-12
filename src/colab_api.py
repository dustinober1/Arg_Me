"""
Google Colab API Client and Utilities.

Interacts with the Google Colaboratory REST API (`colaboratory.googleapis.com`)
using the Google API Client Discovery service and handles Colab environment detection.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from googleapiclient.discovery import build, Resource

from src.config import (
    GOOGLE_CLOUD_PROJECT,
    COLAB_DISCOVERY_URL,
    COLAB_LOCATION,
)



def in_colab() -> bool:
    """Return True if executing inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def get_colab_service(
    project_id: Optional[str] = None,
    discovery_url: str = COLAB_DISCOVERY_URL,
) -> Resource:
    """
    Build and return a Google Colaboratory REST API service client.

    Requires Google Cloud credentials (Application Default Credentials,
    service account file via GOOGLE_APPLICATION_CREDENTIALS, or gcloud auth).

    Args:
        project_id: Optional GCP Project ID. Defaults to env GOOGLE_CLOUD_PROJECT.
        discovery_url: Discovery REST endpoint for Colab API v1.

    Returns:
        Google API client Resource for the Colaboratory service.
    """
    try:
        credentials, discovered_project = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        resolved_project = project_id or GOOGLE_CLOUD_PROJECT or discovered_project

        service = build(
            serviceName="colaboratory",
            version="v1",
            discoveryServiceUrl=discovery_url,
            credentials=credentials,
            cache_discovery=False,
        )
        return service
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "Google Cloud credentials not found. Ensure you have authenticated via:\n"
            "  1. `gcloud auth application-default login`\n"
            "  2. Or set GOOGLE_APPLICATION_CREDENTIALS in .env pointing to your service account key.\n"
            f"Details: {exc}"
        ) from exc


def check_colab_api_status(project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Test connectivity to the Colab REST API endpoint and check project setup.
    """
    status: Dict[str, Any] = {
        "in_colab_runtime": in_colab(),
        "credentials_valid": False,
        "project_id": project_id or GOOGLE_CLOUD_PROJECT,
        "service_built": False,
        "error": None,
    }

    try:
        credentials, discovered_project = default()
        status["credentials_valid"] = True
        status["discovered_project"] = discovered_project
        if not status["project_id"]:
            status["project_id"] = discovered_project

        service = get_colab_service(project_id=status["project_id"])
        status["service_built"] = True
        status["service_name"] = service._serviceName
        status["service_version"] = service._version
    except Exception as exc:
        status["error"] = str(exc)

    return status


if __name__ == "__main__":
    import pprint
    print("Testing Google Colab API configuration...")
    result = check_colab_api_status()
    pprint.pprint(result)
