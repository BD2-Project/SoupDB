"""Deterministic reference/profile -> vector resolver for SQL tests."""


class FakeQueryVectorizer:
    def __init__(self):
        self.vectors = {}
        self.calls = []

    def register(self, reference: str, profile_id: str, vector):
        self.vectors[(reference, profile_id)] = vector

    def vectorize(self, reference: str, *, profile_id: str):
        self.calls.append((reference, profile_id))
        return self.vectors[(reference, profile_id)]
