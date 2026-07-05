from pydantic import BaseModel, Field

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "grade": ["A", "B", "C", "D", "E", "F", "G"],
    "term": ["36 months", "60 months"],
    "loan_purpose": [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "other",
        "major_purchase",
        "medical",
        "small_business",
        "car",
        "vacation",
        "moving",
        "house",
        "wedding",
        "renewable_energy",
        "educational",
    ],
}

SAMPLE_APPLICATION: dict[str, float | int | str] = {
    "loan_amount": 12_000.0,
    "annual_income": 80_000.0,
    "debt_to_income_ratio": 0.2,
    "credit_score": 710,
    "interest_rate": 0.11,
    "installment": 380.0,
    "revol_util": 0.4,
    "grade": "B",
    "term": "36 months",
    "loan_purpose": "home_improvement",
}


class LoanApplicationPayload(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    annual_income: float = Field(..., gt=0, description="Annual income in USD (pre-tax)")
    debt_to_income_ratio: float = Field(
        ..., ge=0, description="Monthly debt payments / monthly gross income"
    )
    credit_score: int = Field(..., ge=300, le=850, description="FICO score (fico_range_low)")
    interest_rate: float = Field(
        ..., ge=0, le=1, description="Annual interest rate as decimal, e.g. 0.12"
    )
    installment: float = Field(..., gt=0, description="Monthly payment amount in USD")
    revol_util: float = Field(
        ..., ge=0, le=1, description="Revolving credit utilisation rate, e.g. 0.55"
    )
    grade: str = Field(..., description="LendingClub loan grade: A, B, C, D, E, F or G")
    term: str = Field(..., description="Loan term: '36 months' or '60 months'")
    loan_purpose: str = Field(
        ..., description="Loan purpose, e.g. debt_consolidation, home_improvement"
    )


def loan_application_schema() -> dict[str, dict]:
    json_schema = LoanApplicationPayload.model_json_schema()
    required = set(json_schema.get("required", []))
    fields: dict[str, dict] = {}

    for name, prop in json_schema["properties"].items():
        meta = {key: value for key, value in prop.items() if key != "title"}
        meta["required"] = name in required
        if name in CATEGORICAL_OPTIONS:
            meta["options"] = CATEGORICAL_OPTIONS[name]
        fields[name] = meta

    return fields
