import httpx
from typing import Optional
from client.core.errors import FrontError
from client.core.logger import logging
from client.core.config import settings
from client.core.session import Session
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

    async def fetch(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json: dict | None = None,
        files: dict | None = None,
        strict: bool = False,
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

        if json and isinstance(json, BaseModel):
            json = json.model_dump(mode="json")

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                files=files,
                headers=headers,
            )
            return self._handle_response(
                response,
                response_model=response_model,
                strict=strict
            )

        except httpx.TimeoutException:
            raise FrontError("REQUEST_TIMEOUT")

        except httpx.ConnectError:
            raise FrontError("SERVICE_UNAVAILABLE")

        except httpx.RequestError as e:
            logging.exception(e)
            raise FrontError("SERVICE_UNAVAILABLE")

    def _handle_response(
            self,
            response: httpx.Response,
            response_model: Optional[BaseModel] = None,
            strict: bool = False
    ):
        if 200 <= response.status_code < 300:
            if response.content:
                try:
                    payload = response.json()
                    if response_model and strict:
                        adapter = TypeAdapter(response_model)
                        return adapter.validate_python(payload)
                    return payload
                except Exception:
                    raise FrontError("INVALID_RESPONSE")
            return None

        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                raise FrontError("INVALID_ERROR_RESPONSE")

            error = payload.get("error")
            if error:
                raise FrontError(error=error)

            raise FrontError("MISSING_ERROR_RESPONSE")
