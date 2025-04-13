import unittest
from agents.general_agent import general_ai_agent

class TestGeneralAIAgent(unittest.TestCase):
    def test_response(self):
        response, confidence = general_ai_agent("What is pathology?")
        self.assertIsInstance(response, str)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

if __name__ == '__main__':
    unittest.main()
