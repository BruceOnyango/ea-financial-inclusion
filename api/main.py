"""JSON prediction service for the financial-inclusion model.

Exposes /health and /predict so an external channel (a Turn.io WhatsApp journey)
can screen one person. Reuses the dashboard's model bundle and prediction logic;
computes no statistics of its own.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field

from fia import artifacts, predict

API_PREFIX = "/financial-inclusion/api"


@lru_cache(maxsize=1)
def get_bundle() -> dict[str, Any]:
    """Load the model bundle once."""
    return artifacts.load_model()


class Respondent(BaseModel):
    """One person's demographic profile (all fields required)."""

    country: str = Field(examples=["Kenya"])
    location_type: str = Field(examples=["Rural"])
    cellphone_access: str = Field(examples=["Yes"])
    household_size: int = Field(examples=[4])
    age_of_respondent: int = Field(examples=[32])
    gender_of_respondent: str = Field(examples=["Female"])
    relationship_with_head: str = Field(examples=["Head of Household"])
    marital_status: str = Field(examples=["Married/Living together"])
    education_level: str = Field(examples=["Primary education"])
    job_type: str = Field(examples=["Self employed"])


# Friendly WhatsApp labels -> exact training categories (WhatsApp list rows cap at 24 chars)
LABEL_ALIASES = {
    "Vocational training": "Vocational/Specialised training",
    "Other/Don't know": "Other/Dont know/RTA",
    "Employed Government": "Formally employed Government",
    "Employed Private": "Formally employed Private",
    "Refuse to answer": "Dont Know/Refuse to answer",
}


def normalize(record: dict[str, Any]) -> dict[str, Any]:
    """Map friendly WhatsApp labels back to exact training categories."""
    return {
        k: LABEL_ALIASES.get(v, v) if isinstance(v, str) else v
        for k, v in record.items()
    }


def recommendation(banked: bool) -> str:
    """Plain next step for a field officer, keyed off the verdict."""
    if banked:
        return (
            "Likely already banked. No inclusion outreach needed; consider "
            "savings or cross-sell products."
        )
    return (
        "Likely unbanked. Prioritise for outreach: account opening, mobile-money "
        "onboarding, or referral to the nearest agent."
    )


def screen(record: dict[str, Any]) -> dict[str, Any]:
    record = normalize(record)
    out = predict.predict_one(get_bundle(), record)
    banked = out["label"] == 1
    return {
        "probability": out["probability"],
        "probability_pct": round(out["probability"] * 100, 1),
        "banked": banked,
        "verdict": out["label_text"],
        "recommendation": recommendation(banked),
    }


def create_api() -> FastAPI:
    api = FastAPI(
        title="Financial Inclusion Prediction API",
        version="1.0.0",
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": get_bundle()["winner"]}

    @router.post("/predict")
    def predict_route(person: Respondent) -> dict[str, Any]:
        return screen(person.model_dump())

    api.include_router(router)
    return api


app = create_api()
