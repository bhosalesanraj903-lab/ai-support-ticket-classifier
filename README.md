# AI support ticket classifier

A FastAPI service that reads a customer support ticket and returns a structured classification: issue category, the team that should own it, priority, customer sentiment, a confidence score, and whether a person should review it. Classification runs on OpenAI `gpt-4o-mini` inside a LangGraph pipeline that redacts personal data, screens for prompt injection, validates the model's output, retries on failure, and logs the cost of every call.

Example request:

```json
{ "ticket_text": "I was charged twice for order #9981. Please refund immediately!", "channel": "web_form" }
```

Response:

```json
{
  "issue_category": "payment_issue",
  "assigned_team": "payments_team",
  "priority": "high",
  "user_sentiment": "angry",
  "confidence_score": 0.95,
  "reasoning": "The customer is reporting a double charge and requesting an immediate refund, indicating a payment issue with high urgency.",
  "requires_human_review": false,
  "pii_detected": false,
  "prompt_version": "v2",
  "cost_info": { "model": "gpt-4o-mini", "input_tokens": 14, "output_tokens": 62, "total_cost_usd": 0.000039 },
  "injection_blocked": false
}
```

## How it works

Each ticket goes through six LangGraph nodes (`graph.py`):

```
pii_redact -> injection_check -> classify -> validate --pass--> cost_log
                                                 \--fail--> fallback --> cost_log
```

1. `pii_redact` replaces email addresses, phone numbers, and credit card numbers with placeholders such as `[EMAIL REDACTED]` before any text reaches the model.
2. `injection_check` asks a separate LLM call whether the ticket is trying to override the system prompt. If it is, the pipeline skips classification and returns a safe default with `injection_blocked: true`. If the guard call itself fails, the ticket is treated as unsafe.
3. `classify` sends the redacted ticket to the model with the active prompt version and JSON-mode output.
4. `validate` checks the result against the Pydantic schema in `schema.py`.
5. `fallback` runs only when validation fails. It retries up to 3 times with exponential backoff (via `tenacity`), then returns a safe classification flagged for human review.
6. `cost_log` counts tokens with `tiktoken` and logs the cost in USD.

Possible values:

| Field | Values |
|---|---|
| `issue_category` | `order_issue`, `payment_issue`, `delivery_issue`, `product_issue`, `account_issue`, `refund_request`, `other` |
| `assigned_team` | `fulfillment_team`, `payments_team`, `logistics_team`, `customer_support`, `tech_team` |
| `priority` | `low`, `medium`, `high`, `critical` |
| `user_sentiment` | `positive`, `neutral`, `negative`, `angry` |

## Project layout

```
main.py                  FastAPI app and endpoints
graph.py                 LangGraph pipeline
schema.py                Pydantic models and enums
production_modules/
  pii_redaction.py       regex-based PII redaction
  prompt_injection.py    LLM-judge injection guard
  structured_output.py   JSON-mode classification call
  validate_response.py   schema validation of model output
  fallback_retry.py      retry with backoff and safe fallback
  non_determinism.py     temperature=0 + seed helpers and a consistency check
  prompt_versioning.py   prompt registry (v1, v2), picked by PROMPT_VERSION
  cost_calculator.py     token counting and per-model pricing
demo_ui/index.html       browser demo page
tests/                   pytest suite
run_manual_tests.py      end-to-end checks against a running server
```

## Setup

Requires Python 3.11 or newer and an OpenAI API key.

```bash
git clone https://github.com/bhosalesanraj903-lab/ai-support-ticket-classifier.git
cd ai-support-ticket-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=sk-...
PROMPT_VERSION=v2
DEFAULT_MODEL=gpt-4o-mini
LOG_COSTS=true
```

`.env` is in `.gitignore`. Don't commit it.

## Running

Start the server:

```bash
python -m uvicorn main:app --port 8000
```

Then use any of these:

- Open `demo_ui/index.html` in a browser, pick a sample ticket or type your own, and click **Classify Ticket**.
- Open http://localhost:8000/docs for the interactive API docs.
- Call the API with curl:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "Tracking says delivered but nothing arrived.", "channel": "web_form"}'
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/classify` | Classify a ticket. Body: `ticket_text` (5 to 4000 characters), `channel` (`web_form` or `email`). |
| `GET` | `/health` | Returns `{"status": "ok"}`. |
| `GET` | `/prompts` | Lists prompt versions and the active one. |

## Testing

Unit tests:

```bash
python -m pytest -v
```

Some tests mock OpenAI and some call it, so a valid key in `.env` is needed for the full suite. `test_injection_safe_ticket` currently fails because its mock patches `.invoke` on the structured LLM, while the code calls it through a `prompt | llm` chain.

End-to-end checks against the running server:

```bash
python run_manual_tests.py
```

This sends 12 tickets covering every category, PII redaction, prompt injection, a vague ticket, and two invalid inputs, then prints a results table and the total cost. A full run costs well under one cent with `gpt-4o-mini`. The category checks depend on the model's judgment, so a borderline ticket can occasionally land in a neighboring category.

## Limitations

- PII redaction is regex-based. It catches emails, phone numbers, and card numbers, but not names or street addresses.
- The injection guard adds one extra LLM call per ticket.
- Prompts live in code (`prompt_versioning.py`) rather than in a database.
