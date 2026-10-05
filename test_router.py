from laya import Router

# Map the local directory to the recognized model alias "laya"
router = Router(models={"laya": "./laya-model"})

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "blocking"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    }
}

result = router.predict(state, questions)

print("Department:", result["answers"]["department"]["choice"])
print("Churn Risk Probability:", result["answers"]["churn_risk"]["noul"])
