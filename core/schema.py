from pydantic import BaseModel, Field


class LoanApplicationPayload(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    annual_income: float = Field(..., gt=0, description="Annual income in USD (pre-tax)")
    debt_to_income_ratio: float = Field(..., ge=0, description="Monthly debt payments / monthly gross income")
    credit_score: int = Field(..., ge=300, le=850, description="FICO score (fico_range_low)")
    interest_rate: float = Field(..., ge=0, le=1, description="Annual interest rate as decimal, e.g. 0.12")
    installment: float = Field(..., gt=0, description="Monthly payment amount in USD")
    revol_util: float = Field(..., ge=0, le=1, description="Revolving credit utilisation rate, e.g. 0.55")
    grade: str = Field(..., description="LendingClub loan grade: A, B, C, D, E, F or G")
    term: str = Field(..., description="Loan term: '36 months' or '60 months'")
    loan_purpose: str = Field(..., description="Loan purpose, e.g. debt_consolidation, home_improvement")
