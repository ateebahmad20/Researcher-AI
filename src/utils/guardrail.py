from guardrails.hub import NSFWText
from guardrails import Guard

guard = Guard().use(
    NSFWText, threshold=0.8, validation_method="sentence", on_fail="exception"
)