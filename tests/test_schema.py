import pytest
from pydantic import ValidationError

from core.schema import LoanApplicationPayload, SAMPLE_APPLICATION, loan_application_schema


def test_valid_application_payload(sample_application):
    payload = LoanApplicationPayload(**sample_application)

    assert payload.loan_amount == 12_000.0
    assert payload.credit_score == 710


def test_loan_application_schema_exports_field_metadata():
    schema = loan_application_schema()

    assert set(schema) == set(SAMPLE_APPLICATION)
    assert schema["credit_score"]["type"] == "integer"
    assert schema["credit_score"]["minimum"] == 300
    assert schema["loan_amount"]["exclusiveMinimum"] == 0
    assert "debt_consolidation" in schema["loan_purpose"]["options"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("loan_amount", 0),
        ("annual_income", -1),
        ("credit_score", 250),
        ("credit_score", 900),
        ("interest_rate", 1.5),
        ("revol_util", 1.2),
    ],
)
def test_invalid_application_payload_rejected(sample_application, field, value):
    invalid = sample_application | {field: value}

    with pytest.raises(ValidationError):
        LoanApplicationPayload(**invalid)
