import main


class _FakeItem:
    def __init__(self, value):
        self.value = value


class _FakeStore:
    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value, index=None):
        self.data[(namespace, key)] = value

    def get(self, namespace, key):
        value = self.data.get((namespace, key))
        if value is None:
            return None
        return _FakeItem(value)


def test_extract_persona_instruction_recognizes_role_assignment():
    text = "你现在是我的私人助理，负责处理我的私人事务"
    assert main._extract_persona_instruction(text) == text


def test_save_and_load_persona_instruction(monkeypatch):
    fake_store = _FakeStore()
    monkeypatch.setattr(main, "store", fake_store)

    text = "你现在是我的私人助理，负责处理我的私人事务"
    saved = main._save_persona_instruction(text, "u1")

    assert saved == text
    assert main._load_persona_instruction("u1") == text


def test_build_persona_system_instruction_contains_original_instruction():
    text = "你现在是我的私人助理，负责处理我的私人事务"
    rendered = main._build_persona_system_instruction(text)

    assert "Persistent user role instruction" in rendered
    assert text in rendered
