# Test Plan & Report

## Strategy

Two test levels: 5 unit tests (no services needed) and integration tests (FastAPI running — 8 tests without model and DB, 3 test requiring model-server and DB). Tests run automatically via GitHub Actions CI/CD on every model deployment.

---

## Acceptance Criteria

| Criteria | Threshold | Result |
|----------|-----------|--------|
| All unit tests pass | 5/5 | Pass |
| All integration tests pass | 10/10 | Pass |
| /ready returns 200 when deployed | 200 OK | Pass |
| Model F1 weighted (Default Threshold) | > 0.70 | Pass (0.92) |

---

## Test Cases

| ID | Level | Description | Input | Expected | Result |
|----|-------|-------------|-------|----------|--------|
| U1 | Unit | Output shape always 1101 points | Valid spectrum | shape == (1101,) | Pass |
| U2 | Unit | NaN values handled gracefully | Spectrum with NaN | No NaN in output | Pass |
| U3 | Unit | Empty spectrum raises error | Empty arrays | ValueError raised | Pass |
| U4 | Unit | Basic vs advanced preprocessing differ | Same spectrum both modes | Outputs not equal | Pass |
| U5 | Unit | DriftDetector returns correct keys | Valid spectrum | keys: drift_score, is_drifted | Pass |
| I1 | Integration | GET /health returns 200 | — | 200, status=ok | Pass |
| I2 | Integration | GET /ready returns 200 | — | 200 (model up) | Pass |
| I3 | Integration | GET /metrics contains counters | — | 200, predictions_total present | Pass |
| I4 | Integration | GET /classes returns 5 classes | — | 200, 5 garnet classes | Pass |
| I5 | Integration | POST /predict wrong extension | .csv file | 400 | Pass |
| I6 | Integration | POST /predict empty txt | Empty file | 400/422 | Pass |
| I7 | Integration | POST /predict zip no txt | Zip without .txt | 400 | Pass |
| I8 | Integration | POST /predict corrupted data | Junk data | 400/422 | Pass |
| I9 | Integration | POST /feedback unknown file | Unknown filename | 404 | Pass |
| I10 | Integration | POST /feedback missing field | Incomplete body | 422 | Pass |
| I11 | Integration | POST /predict valid spectrum dry run | Real `.txt` spectrum file | 200, predicted_class returned | Pass |

---

## Results

| Category | Total | Passed | Failed |
|----------|-------|--------|--------|
| Unit | 5 | 5 | 0 |
| Integration | 11 | 11 | 0 |
| **Total** | **16** | **16** | **0** |

**All acceptance criteria met. Pass rate: 100% (16/16)**

---

## How To Run

```bash
# Unit tests
python -m pytest tests/test_preprocessor.py -v

# Integration tests (FastAPI running)
API_URL=http://localhost:8000 python -m pytest tests/test_api.py -v -m "not needs_model"

# Full system tests (all services(model, feedback) running)
API_URL=http://localhost:8000 python -m pytest tests/test_api.py -v
```