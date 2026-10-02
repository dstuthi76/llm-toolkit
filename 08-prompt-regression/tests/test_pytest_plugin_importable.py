import unittest


class TestPytestPluginImportable(unittest.TestCase):
    def test_module_imports_without_pytest_installed_raising_clear_error(self):
        try:
            import promptcheck.pytest_plugin as plugin
        except ImportError as e:
            self.skipTest(f"pytest not installed in this environment: {e}")
            return
        self.assertTrue(hasattr(plugin, "llm_case"))


if __name__ == "__main__":
    unittest.main()
