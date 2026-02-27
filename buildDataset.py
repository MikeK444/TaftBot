import json
from pathlib import Path
import random

RAWTEXTDIRECTORY = Path("rawTexts")
OUTPUTFILE = "taftInstruct.jsonl"

MINWORDS = 120
MAXWORD = 350
MINKEYWORDHITS = 2

QUESTIONTEMPLATES = [
    "what are your view on {topic}",
    "How did you approach the issue of {topic}?",
    "Why was {topic} important during your presidency",
    "Can you explain your perspective on {topic}",
    "What principles guided your thinking about {topic}"
]

TOPIC_KEYWORDS = {
    "the judiciary": ["court", "judicial", "constitution", "law"],
    "foreign policy": ["foreign", "arbitration", "treaty"],
    "legislation": ["congress", "legislation"],
    "economic development": ["economy", "industry", "commerce"],
    "race relations": ["south", "race", "reconstruction"],
    "executive power": ["presidency", "executive"]
}


def semanticChunks(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_words = []
    for para in paragraphs:
        words = para.split()
        if len(current_words) + len(words) <= MAXWORD:
            current_words.extend(words)
        else:
            if len(current_words) >= MINWORDS:
                chunks.append(" ".join(current_words))
            current_words = words
    if len(current_words) >= MINWORDS:
        chunks.append(" ".join(current_words))
    return chunks

def findRelavantTopic(chunk):
    lower = chunk.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = sum(lower.count(k) for k in keywords)
        if hits >= MINKEYWORDHITS:
            return topic, keywords
    return None, None
def extractCoreSentences(chunk, keywords, maxSentences=2):
    sentences = [s.strip() for s in chunk.split(".") if s.strip()]
    relevant = [s for s in sentences if any(k in s.lower() for k in keywords)]
    if not relevant:
        return None
    core = ". ".join(relevant[:maxSentences])
    return core + "."


ANSWER_STYLES = [
    # Explanatory
    lambda topic, core: (
        f"My views on {topic} were grounded in constitutional principle. "
        f"{core} I believed restraint and respect for law were essential."
    ),

    # Reflective
    lambda topic, core: (
        f"In reflecting upon my public service, {core} "
        f"This belief guided my conduct as President."
    ),

    # Justificatory
    lambda topic, core: (
        f"I believed it necessary to approach {topic} with deliberation and care. "
        f"{core} Such judgment best served the nation."
    ),

    # Defensive
    lambda topic, core: (
        f"Some critics misunderstood my position on {topic}, but {core} "
        f"My actions were guided by duty rather than ambition."
    ),

    # Educational
    lambda topic, core: (
        f"To understand my position on {topic}, one must consider the constitutional framework. "
        f"{core} This context is essential to sound judgment."
    )
]

def generateAnswer(chunk, topic, keywords):
    core = extractCoreSentences(chunk, keywords, 2)
    if core is None:
        return None
    style = random.choice(ANSWER_STYLES)
    return style(topic, core)


examples = []

for file_path in RAWTEXTDIRECTORY.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")
    chunks = semanticChunks(text)
    for chunk in chunks:
        topic, keywords = findRelavantTopic(chunk)
        if topic is None:
            continue
        templates = random.sample(QUESTIONTEMPLATES, k=3)
        for template in templates:
            output = generateAnswer(chunk, topic, keywords)
            if output is None:
                continue
            examples.append({
                "instruction": template.format(topic=topic),
                "input": "",
                "output": output
            })

with open(OUTPUTFILE, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} training examples.")