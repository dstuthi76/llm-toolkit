"""
Example: a toy "book a flight" agent that fails partway through, then
tracewright diagnoses exactly which step caused the failure.

Run: python3 example.py
Then open tracewright_report.html in a browser.
"""
from tracewright import Tracer, Judge, generate_html_report

tracer = Tracer(task="Book a round-trip flight to Paris under $800")


@tracer.trace_tool("check_calendar")
def check_calendar():
    return {"free_dates": ["2026-11-03", "2026-11-10"]}


@tracer.trace_tool("search_flights")
def search_flights(destination, date):
    # Simulate a flaky external API returning nothing useful.
    return []


@tracer.trace_llm("gpt-4o-mini")
def pick_best_flight(candidates):
    if not candidates:
        return "I could not find any flights matching your criteria."
    return f"Booking the cheapest option: {candidates[0]}"


# --- run the "agent" ---
calendar = check_calendar()
flights = search_flights("Paris", calendar["free_dates"][0])
final_answer = pick_best_flight(flights)

print("Agent's final answer:", final_answer)

# --- diagnose the run ---
diagnosis = Judge().diagnose(tracer.trace)   # MockLLMClient by default, no API key needed
print(f"\nVerdict: {diagnosis.verdict}")
print(f"Root cause step: {diagnosis.root_cause_step}")
print(f"Reason: {diagnosis.reason}")

with open("tracewright_report.html", "w", encoding="utf-8") as f:
    f.write(generate_html_report(tracer.trace, diagnosis))
print("\nWrote tracewright_report.html")
