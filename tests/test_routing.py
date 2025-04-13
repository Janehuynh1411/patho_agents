def test_general_ai_response():
    from agents.general_agent import general_ai_agent
    response, conf = general_ai_agent("What is pathology?")
    assert isinstance(response, str)
    assert 0 <= conf <= 1
