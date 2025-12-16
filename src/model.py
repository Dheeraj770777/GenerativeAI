from sklearn.ensemble import RandomForestClassifier


def build_model(n_estimators: int = 100, random_state: int = 42):
    """Create and return a RandomForestClassifier configured with given params."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
