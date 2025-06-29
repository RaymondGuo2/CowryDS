import statsmodels.api as sm

def forward_selection(X, y, criterion='aic'):
    initial_features = []
    remaining_features = list(X.columns)
    best_score = float('inf')
    best_model = None

    while remaining_features:
        scores_with_candidates = []

        for candidate in remaining_features:
            features = initial_features + [candidate]
            X_model = sm.add_constant(X[features])
            model = sm.OLS(y, X_model).fit()
            score = getattr(model, criterion)
            scores_with_candidates.append((score, candidate, model))

        scores_with_candidates.sort()
        best_candidate_score, best_candidate, candidate_model = scores_with_candidates[0]

        if best_candidate_score < best_score:
            best_score = best_candidate_score
            best_model = candidate_model
            initial_features.append(best_candidate)
            remaining_features.remove(best_candidate)
        else:
            break

    return best_model, initial_features