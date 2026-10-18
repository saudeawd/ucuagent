"""Deterministic checks over the extracted case documents.

Small hosted models (Qwen2.5-7B, gpt-oss-20b) are unreliable at arithmetic
across several documents. These checks compute the facts exactly and hand
them to the decision prompt as `rule_checks`; the model still decides.

Every check is generic: it relies on document conventions (position lines
"Menge  EP  GP", "Summe netto", award notes listing netto/brutto per offer,
form checkboxes), never on case ids or known amounts. A check that cannot
find its inputs reports nothing rather than guessing.

Each finding is {"check", "severity", "finding"} where severity is
"verstoss" (a material, document-backed violation unless another document
explicitly resolves it), "hinweis" (needs the model's judgment), or "ok".
"""
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

AMOUNT = r"\d{1,3}(?:\.\d{3})*,\d{2}"
AMOUNT_PATTERN = re.compile(rf"(?<![\d.,])({AMOUNT})(?![\d,])")
# "3  740,00 €  2.220,00 €" at the end of a position line: quantity, unit and
# total price. Only lines where quantity * unit == total count as positions,
# so headers, dates, and phone numbers can never be mistaken for one.
POSITION_PATTERN = re.compile(
    rf"(?<![\d.,])(\d{{1,4}})\s+(?:x\s+|Stk\.?\s+|St\.?\s+)?({AMOUNT})\s*(?:€|EUR)?\s+({AMOUNT})\s*(?:€|EUR)?\s*$"
)
NET_TOTAL_PATTERN = re.compile(
    r"(summe|gesamt|zwischensumme|total)[^\n]*netto|netto[^\n]*(summe|gesamt)|nettobetrag",
    re.IGNORECASE,
)
VAT_RATES = (Decimal("0.19"), Decimal("0.07"))
MATERIAL_RELATIVE_DIFFERENCE = Decimal("0.05")
DIRECT_AWARD_THRESHOLD = Decimal("1000")
UNIVERSITY_MARKERS = re.compile(
    r"universit|hochschule|lehrstuhl|fachbereich|fachgebiet|institut|fakult|professur|"
    r"kostenstelle|dienstlich|labor|projekt|versuch|seminar|konferenz|conference|tagung|workshop|"
    r"forschung|lehrveranstaltung",
    re.IGNORECASE,
)
ORDER_NUMBER_PATTERN = re.compile(
    r"(?:bestell(?:ung|nummer|-nr\.?)|order(?:\s+(?:no\.?|number|id))?|auftrags?(?:nummer|-nr\.?))"
    r"[:#\s]*([A-Z0-9][A-Z0-9\-/]{4,})",
    re.IGNORECASE,
)


def parse_amount(text: str) -> Optional[Decimal]:
    try:
        return Decimal(text.replace(".", "").replace(",", "."))
    except InvalidOperation:
        return None


def format_amount(value: Decimal) -> str:
    whole, cents = f"{value:.2f}".split(".")
    return f"{int(whole):,}".replace(",", ".") + f",{cents} €"


def amounts_in(line: str) -> list[Decimal]:
    return [value for value in map(parse_amount, AMOUNT_PATTERN.findall(line)) if value is not None]


def relative_difference(a: Decimal, b: Decimal) -> Decimal:
    return abs(a - b) / max(abs(a), abs(b), Decimal("0.01"))


def position_totals(text: str) -> list[Decimal]:
    totals = []
    for line in text.splitlines():
        match = POSITION_PATTERN.search(line.strip())
        if not match:
            continue
        quantity, unit, total = int(match.group(1)), parse_amount(match.group(2)), parse_amount(match.group(3))
        if unit is not None and total is not None and quantity > 0 and abs(quantity * unit - total) <= Decimal("0.01"):
            totals.append(total)
    return totals


def stated_net_total(text: str) -> Optional[Decimal]:
    """The last amount on the first line that labels a net total."""
    for line in text.splitlines():
        if NET_TOTAL_PATTERN.search(line):
            values = amounts_in(line)
            if values:
                return values[-1]
    return None


def check_position_sums(documents: dict[str, str]) -> list[dict[str, str]]:
    findings = []
    for filename, text in documents.items():
        positions = position_totals(text)
        stated = stated_net_total(text)
        if not positions or stated is None:
            continue
        computed = sum(positions, Decimal("0"))
        if relative_difference(computed, stated) > MATERIAL_RELATIVE_DIFFERENCE:
            findings.append({
                "check": "positionssumme",
                "severity": "verstoss",
                "finding": (
                    f"{filename}: Die Einzelpositionen ergeben {format_amount(computed)} netto, das Dokument "
                    f"nennt aber {format_amount(stated)} netto (Abweichung "
                    f"{relative_difference(computed, stated) * 100:.0f} %)."
                ),
            })
        else:
            findings.append({
                "check": "positionssumme",
                "severity": "ok",
                "finding": f"{filename}: Summe der Positionen {format_amount(computed)} stimmt mit der genannten Nettosumme überein.",
            })
    return findings


NOT_AN_OFFER = ("rechnung", "invoice", "email", "e-mail", "kontoauszug", "warenkorb", "korrespondenz")


def is_invoice(filename: str) -> bool:
    return any(word in filename.casefold() for word in ("rechnung", "invoice"))


def is_offer(filename: str, text: str) -> bool:
    name = filename.casefold()
    if any(word in name for word in NOT_AN_OFFER):
        return False
    head = "\n".join(text.splitlines()[:25])
    return "angebot" in name or bool(re.search(r"^\s*angebot\b", head, re.IGNORECASE | re.MULTILINE))


def is_award_note(filename: str, text: str) -> bool:
    return "vergabevermerk" in filename.casefold() or text.lstrip().casefold().startswith("vergabevermerk")


def net_brutto_rows(text: str) -> list[tuple[str, Decimal]]:
    """Rows such as "Bieter  Nr.  vom  2.600,00 €  3.094,00 €  1" in an
    award note: a net amount directly followed by its gross amount."""
    rows = []
    for line in text.splitlines():
        values = amounts_in(line)
        for net, gross in zip(values, values[1:]):
            if any(abs(net * (1 + rate) - gross) <= Decimal("0.02") for rate in VAT_RATES):
                rows.append((line.strip(), net))
                break
    return rows


def check_procurement(documents: dict[str, str]) -> list[dict[str, str]]:
    findings = []
    offers = {
        filename: stated_net_total(text)
        for filename, text in documents.items()
        if is_offer(filename, text) and not is_award_note(filename, text)
    }
    offer_nets = {filename: net for filename, net in offers.items() if net is not None}
    joined = "\n".join(documents.values())

    framework = re.search(r"rahmenvertrag\w*\s+\[\s*x\s*\]\s*ja", joined, re.IGNORECASE)
    no_framework = re.search(r"rahmenvertrag\w*\s+\[\s*\]\s*ja\s+\[\s*x\s*\]\s*nein", joined, re.IGNORECASE)
    if framework:
        findings.append({"check": "rahmenvertrag", "severity": "hinweis",
                         "finding": "Das Formular kreuzt 'Rahmenvertragsprodukt: ja' an; dann gilt die Bestellung beim Rahmenvertragspartner statt Vergleichsangeboten."})
    elif no_framework:
        findings.append({"check": "rahmenvertrag", "severity": "hinweis",
                         "finding": "Das Formular kreuzt 'Rahmenvertragsprodukt: nein' an; prüfe trotzdem, ob die Produktgruppe unter einen Rahmenvertrag fällt."})

    # Order value: the net total of the request form, else the cheapest offer.
    order_net = None
    for filename, text in documents.items():
        if not is_offer(filename, text) and not is_award_note(filename, text) and "antrag" in filename.casefold():
            order_net = stated_net_total(text) or order_net
    if order_net is None and offer_nets:
        order_net = min(offer_nets.values())

    if order_net is not None:
        count = len(offers)
        stated_count = re.search(r"eingeholte\s+angebote\s+(\d+)", joined, re.IGNORECASE)
        detail = f"{count} Angebotsdokument(e) liegen vor"
        if stated_count:
            detail += f", das Formular nennt {stated_count.group(1)} eingeholte Angebote"
        if order_net < DIRECT_AWARD_THRESHOLD:
            findings.append({"check": "vergleichsangebote", "severity": "ok",
                             "finding": f"Auftragswert {format_amount(order_net)} netto liegt unter 1.000 € netto: Direktvergabe zulässig, {detail}."})
        elif count < 3 and not framework:
            findings.append({"check": "vergleichsangebote", "severity": "verstoss",
                             "finding": (f"Auftragswert {format_amount(order_net)} netto ab 1.000 €: {detail}, gefordert sind "
                                         "drei. Verstoß, sofern kein Rahmenvertrag greift und keine nachvollziehbare Begründung "
                                         "für fehlende Angebote dokumentiert ist.")})
        else:
            findings.append({"check": "vergleichsangebote", "severity": "ok",
                             "finding": f"Auftragswert {format_amount(order_net)} netto; {detail}."})

    for filename, text in documents.items():
        if not is_award_note(filename, text) or not offer_nets:
            continue
        for row, net in net_brutto_rows(text):
            nearest = min(offer_nets.items(), key=lambda item: abs(item[1] - net))
            if relative_difference(nearest[1], net) > Decimal("0.005"):
                findings.append({"check": "vergabevermerk_betraege", "severity": "verstoss",
                                 "finding": (f"{filename} nennt {format_amount(net)} netto ('{row[:90]}'), aber kein "
                                             f"vorgelegtes Angebot hat diesen Betrag; nächstliegend {nearest[0]} mit "
                                             f"{format_amount(nearest[1])} netto.")})
        award = re.search(rf"zuschlag[^\n]*?({AMOUNT})", text, re.IGNORECASE)
        if award:
            awarded = parse_amount(award.group(1))
            cheapest_file, cheapest = min(offer_nets.items(), key=lambda item: item[1])
            if awarded is not None and cheapest < awarded and relative_difference(cheapest, awarded) > Decimal("0.005"):
                findings.append({"check": "guenstigstes_angebot", "severity": "verstoss",
                                 "finding": (f"Zuschlag über {format_amount(awarded)} netto, aber {cheapest_file} ist mit "
                                             f"{format_amount(cheapest)} netto günstiger (laut Angebotsdokument).")})
    invoice_nets = [
        net for filename, text in documents.items()
        if is_invoice(filename) and (net := stated_net_total(text)) is not None
    ]
    if invoice_nets and offer_nets:
        invoiced = sum(invoice_nets, Decimal("0"))
        matching = [name for name, net in offer_nets.items() if relative_difference(net, invoiced) <= Decimal("0.005")]
        findings.append({
            "check": "rechnungen_vs_angebot",
            "severity": "ok" if matching else "hinweis",
            "finding": (f"{len(invoice_nets)} Rechnung(en) über zusammen {format_amount(invoiced)} netto"
                        + (f" entsprechen {', '.join(matching)}." if matching else "; kein Angebot hat genau diese Summe.")),
        })
    if offer_nets:
        cheapest_file, cheapest = min(offer_nets.items(), key=lambda item: item[1])
        listing = ", ".join(f"{name}: {format_amount(net)}" for name, net in sorted(offer_nets.items()))
        findings.append({"check": "angebotssummen", "severity": "ok",
                         "finding": f"Nettosummen laut Angebotsdokumenten: {listing}; günstigstes: {cheapest_file}."})
    return findings


def check_expense(documents: dict[str, str]) -> list[dict[str, str]]:
    findings = []
    joined = "\n".join(documents.values())
    if not UNIVERSITY_MARKERS.search(joined):
        findings.append({"check": "dienstlicher_bezug", "severity": "verstoss",
                         "finding": ("Kein Beleg nennt eine Universität, Einrichtung, Kostenstelle oder einen dienstlichen "
                                     "Zweck: der dienstliche Bezug ist nirgends erkennbar.")})
    else:
        findings.append({"check": "dienstlicher_bezug", "severity": "hinweis",
                         "finding": "Mindestens ein Beleg nennt eine Universität/Einrichtung oder einen dienstlichen Bezug."})

    order_numbers: dict[str, set[str]] = {}
    for filename, text in documents.items():
        for number in ORDER_NUMBER_PATTERN.findall(text):
            if any(ch.isdigit() for ch in number):
                order_numbers.setdefault(number.upper(), set()).add(filename)
    for number, filenames in order_numbers.items():
        if len(filenames) > 1:
            invoices = [name for name in filenames if is_invoice(name)]
            findings.append({"check": "doppelte_bestellung",
                             "severity": "verstoss" if len(invoices) > 1 else "hinweis",
                             "finding": (f"Bestellnummer {number} erscheint in {', '.join(sorted(filenames))}"
                                         + (": mehrere Rechnungen zu derselben Bestellung, mögliche Doppelabrechnung; "
                                            "prüfe, ob es Teillieferungen mit unterschiedlichen Positionen sind."
                                            if len(invoices) > 1 else
                                            ": Rechnung und Zahlungsnachweis zur selben Bestellung sind zulässig."))})
    return findings


def run_rule_checks(case_type: str, documents: dict[str, str]) -> list[dict[str, Any]]:
    findings = check_position_sums(documents)
    if case_type == "procurement":
        findings += check_procurement(documents)
    elif case_type == "expense_reimbursement":
        findings += check_expense(documents)
    return findings
