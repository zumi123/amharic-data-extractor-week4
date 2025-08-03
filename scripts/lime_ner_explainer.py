from lime.lime_text import LimeTextExplainer
from transformers import pipeline

# Load fine-tuned NER pipeline
ner = pipeline(
    "token-classification",
    model="models/amharic-ner-afroxlmr",
    tokenizer="models/amharic-ner-afroxlmr",
    aggregation_strategy="simple"
)

# Sample input
text = "ዋጋ 200 ብር ያለው የፀጉር መንጠር ቦታ አዲስ አበባ"

# Classifier wrapper for LIME (binary token classifier approximation)
def classifier_fn(texts):
    outputs = []
    for t in texts:
        result = ner(t)
        output = [0] * len(t.split())  # 1 for important tokens
        for r in result:
            start = r["start"]
            end = r["end"]
            token = t[r["start"]:r["end"]]
            if r["entity_group"] != "O":
                for i in range(len(t.split())):
                    if token in t.split()[i]:
                        output[i] = 1
        outputs.append([1 - output[i] for i in range(len(output))])  # fake probabilities
    return outputs

# Explain
explainer = LimeTextExplainer(class_names=["Non-Entity", "Entity"])
exp = explainer.explain_instance(text, classifier_fn, num_features=10)

# Display
exp.show_in_notebook()
