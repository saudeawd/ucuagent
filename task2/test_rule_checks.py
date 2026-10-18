import json
import os
import unittest
from decimal import Decimal
from pathlib import Path

from rule_checks import (
    check_position_sums,
    format_amount,
    net_brutto_rows,
    position_totals,
    run_rule_checks,
    stated_net_total,
)

DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", Path(__file__).parent / "datasets")).resolve()
INPUTS = {
    "dienstreiseantrag": DATASETS_ROOT / "business-trip-spot-check-inputs",
    "beschaffungsantrag": DATASETS_ROOT / "task-2-procurement-spot-check-inputs",
    "auslagenerstattung": DATASETS_ROOT / "task-2-expense-reimbursement-spot-check-inputs",
}


def gold_labels() -> dict[str, str]:
    labels = {}
    for path in DATASETS_ROOT.glob("*-truths/ground-truth.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                labels[entry["antrag"]] = entry["result"]
    return labels


def violations(case_id: str) -> list[str]:
    import predict_v2

    evidence = predict_v2.build_case_evidence(INPUTS[case_id.rsplit("-", 1)[0]], case_id)
    return [finding["check"] for finding in evidence["rule_checks"] if finding["severity"] == "verstoss"]


class ParserTest(unittest.TestCase):
    def test_counts_only_self_consistent_position_lines(self):
        text = (
            "Pos  Artikel  Menge  EP netto  GP netto\n"
            "1  Netzgerät  3  740,00 €  2.220,00 €\n"
            "Telefon 0561 7390-4702\n"
            "2  Kabel  3  60,00 €  180,00 €\n"
            "3  Falsch  2  10,00 €  99,00 €\n"
        )
        self.assertEqual([Decimal("2220.00"), Decimal("180.00")], position_totals(text))

    def test_reads_the_labelled_net_total(self):
        text = "Summe netto  2.400,00 €\nzzgl. 19 % USt.  456,00 €\nAngebotssumme brutto  2.856,00 €"
        self.assertEqual(Decimal("2400.00"), stated_net_total(text))

    def test_flags_a_material_mismatch_between_positions_and_total(self):
        text = "1  Gerät  3  740,00 €  2.220,00 €\n2  Modul  3  60,00 €  180,00 €\nSumme netto  3.384,00 €"
        findings = check_position_sums({"antrag.pdf": text})
        self.assertEqual("verstoss", findings[0]["severity"])

    def test_accepts_matching_totals(self):
        text = "1  Gerät  3  740,00 €  2.220,00 €\nSumme netto  2.220,00 €"
        self.assertEqual("ok", check_position_sums({"antrag.pdf": text})[0]["severity"])

    def test_finds_net_gross_pairs(self):
        rows = net_brutto_rows("Bieter A  TR-1  02.08.2027  2.600,00 €  3.094,00 €  1")
        self.assertEqual(Decimal("2600.00"), rows[0][1])

    def test_formats_german_amounts(self):
        self.assertEqual("2.400,00 €", format_amount(Decimal("2400")))

    def test_unknown_case_type_runs_only_generic_checks(self):
        self.assertEqual([], run_rule_checks("business_trip", {"a.pdf": "kein Betrag"}))


@unittest.skipUnless(INPUTS["beschaffungsantrag"].is_dir(), "spot-check datasets not available")
class SpotCheckTest(unittest.TestCase):
    def test_flags_inflated_form_total(self):
        self.assertIn("positionssumme", violations("beschaffungsantrag-04"))

    def test_flags_too_few_offers_above_threshold(self):
        self.assertIn("vergleichsangebote", violations("beschaffungsantrag-06"))

    def test_flags_award_note_amount_not_backed_by_offer(self):
        found = violations("beschaffungsantrag-07")
        self.assertIn("vergabevermerk_betraege", found)
        self.assertIn("guenstigstes_angebot", found)

    def test_flags_invoice_without_any_official_reference(self):
        self.assertIn("dienstlicher_bezug", violations("auslagenerstattung-03"))

    def test_flags_two_invoices_for_one_order(self):
        self.assertIn("doppelte_bestellung", violations("auslagenerstattung-06"))

    def test_raises_no_violation_on_accepted_cases(self):
        for case_id, label in gold_labels().items():
            if label == "angenommen":
                with self.subTest(case_id=case_id):
                    self.assertEqual([], violations(case_id))


class PromptTest(unittest.TestCase):
    def test_v2_prompt_explains_rule_checks_and_keeps_json_format(self):
        import predict_v2

        prompt = predict_v2.system_prompt_for("procurement")
        self.assertIn("rule_checks", prompt)
        self.assertTrue(prompt.rstrip().endswith('"begruendung":"kurze belegte Begründung"}'))

    def test_v2_is_wired_into_the_shared_pipeline(self):
        import predict
        import predict_v2

        self.assertIs(predict.build_case_evidence, predict_v2.build_case_evidence)
        self.assertIs(predict.system_prompt_for, predict_v2.system_prompt_for)
