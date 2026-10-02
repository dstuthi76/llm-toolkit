import unittest
from privacyrag.pii import PIIRedactor


class TestPIIRedactor(unittest.TestCase):
    def test_redacts_email(self):
        r = PIIRedactor()
        result = r.redact("Contact John Smith at john.smith@example.com for details.")
        self.assertIn("EMAIL", result.entities_found)
        self.assertNotIn("john.smith@example.com", result.redacted)

    def test_redacts_ssn(self):
        r = PIIRedactor()
        result = r.redact("Patient SSN: 123-45-6789")
        self.assertIn("SSN", result.entities_found)
        self.assertNotIn("123-45-6789", result.redacted)

    def test_redacts_phone(self):
        r = PIIRedactor()
        result = r.redact("Call me at 555-123-4567 tomorrow.")
        self.assertIn("PHONE", result.entities_found)
        self.assertNotIn("555-123-4567", result.redacted)

    def test_redacts_medical_record_number(self):
        r = PIIRedactor()
        result = r.redact("Patient MRN-882910 was admitted yesterday.")
        self.assertIn("MEDICAL_RECORD_NUMBER", result.entities_found)

    def test_no_pii_found_leaves_text_unchanged(self):
        r = PIIRedactor()
        text = "The weather today is sunny with a light breeze."
        result = r.redact(text)
        self.assertEqual(result.entities_found, [])
        self.assertEqual(result.redacted, text)

    def test_can_restrict_to_specific_entity_types(self):
        r = PIIRedactor(entity_types=["EMAIL"])
        result = r.redact("Email jane@example.com or call 555-123-4567.")
        self.assertIn("EMAIL", result.entities_found)
        self.assertNotIn("PHONE", result.entities_found)
        self.assertIn("555-123-4567", result.redacted)  # untouched, we only asked for EMAIL


if __name__ == "__main__":
    unittest.main()
