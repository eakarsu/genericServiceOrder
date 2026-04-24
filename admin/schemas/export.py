from pydantic import BaseModel


class ExportRequest(BaseModel):
    sector: str | None = None
    status: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    search: str | None = None
