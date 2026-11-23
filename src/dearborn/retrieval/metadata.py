"""Deterministic source labels for passage text; never infer query metadata."""

def enrich_passage(text, sources, names):
    if not sources:raise ValueError("Passage source labels are required")
    lines=[]
    for ticker,year,form in sorted(set(sources)):
        name=names.get(ticker,ticker)
        issuer=f"{name} ({ticker})" if name!=ticker else ticker
        lines.append(f"Issuer: {issuer}; document year: {year}; form: {form}.")
    # All observed labels are preserved when a deduplicated text has several sources.
    return "Observed source metadata:\n"+"\n".join(lines)+"\nPassage:\n"+text
