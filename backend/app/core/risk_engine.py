def calculate_risk(probability: float):
    """
    Convert model probability to business risk score, risk level, and decision.
    """
    risk_score = probability * 100

    if probability < 0.15:
        return risk_score, "Low", "Approve"
    elif probability < 0.35:
        return risk_score, "Medium", "Manual Review"
    else:
        return risk_score, "High", "Block Transaction"
