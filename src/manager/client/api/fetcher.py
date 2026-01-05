import httpx
from typing import Optional
from .errors import FrontError
from .logger import logging
from .config import settings
from .session import Session
from pydantic import BaseModel, TypeAdapter


class APIFetcher:
    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        session: Session,
        timeout: float = settings.DEFAULT_TIMEOUT,
        strict: bool = False
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = client
        self.session = session
        self.strict = strict

    async def fetch(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json: dict | None = None,
        retries: int = settings.DEFAULT_RETRIES,
        require_auth: bool = True,
        response_model: Optional[BaseModel] = None
    ):
        url = f"{self.base_url}{path}"
        logging.info(url)

        headers = {}
        if require_auth and self.session.is_authenticated():
            token = self.session.get_token()
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers,
            )
            return self._handle_response(response, response_model=response_model)

        except httpx.TimeoutException:
            raise FrontError("REQUEST_TIMEOUT")

        except httpx.ConnectError:
            raise FrontError("SERVICE_UNAVAILABLE")

        # TODO: add logging here later
        except httpx.RequestError as e:
            logging.exception(e)
            raise FrontError("SERVICE_UNAVAILABLE")

    def _handle_response(
            self,
            response: httpx.Response,
            response_model: Optional[BaseModel] = None
    ):
        if 200 <= response.status_code < 300:
            if response.content:
                try:
                    payload = response.json()
                    if response_model and self.strict:
                        adapter = TypeAdapter(response_model)
                        return adapter.validate_python(payload)

                except Exception:
                    raise FrontError("INVALID_RESPONSE")
            return None

        if response.status_code >= 400:
            try:
                payload = response.json()
                error = payload.get("error")
                if error:
                    raise FrontError(error=error)
                raise FrontError("MISSING_ERROR_RESPONSE")
            except Exception:
                raise FrontError("INVALID_ERROR_RESPONSE")
