from openai import OpenAI
import json
from datasets import load_dataset

def main():
    # DeepSeek Reasoner API setup
    client = OpenAI(api_key="sk-4d5824bb72d54528bd45943def4688db", base_url="https://api.deepseek.com")

    # Randomly load 20 MMLU samples
    dataset = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42).select(range(20))
    rewritten_data = []

    for item in dataset:
        question = item["question"]
        choices = item["choices"]

        original = (
            "Rewrite the following multiple-choice question and answer options."
            "Don't change the original meaning. Do not answer it.\n\n"
            f"Question: {question}\n"
            f"Choices:\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            "Respond in this format:\n"
            "Question: <rewritten question>\n"
            "Choices:\nA. <option A>\nB. <option B>\nC. <option C>\nD. <option D>"
        )

        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "Rewrite question to make it more understandable, just give me the rewritten question: "},
                    {"role": "user", "content": original}
                ],
                stream=True
                # temperature=0.7,
                # max_tokens=256,
            )

            full_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content

            lines = full_response.strip().splitlines()
            rewritten_question = ""
            rewritten_choices = []

            for line in lines:
                if line.lower().startswith("question:"):
                    rewritten_question = line.split(":", 1)[1].strip()
                elif line.strip().startswith(tuple("ABCD")):
                    if '.' in line:
                        rewritten_choices.append(line.split('.', 1)[1].strip())

            if len(rewritten_choices) == 4:
                rewritten_data.append({
                    "original_question": question,
                    "original_choices": choices,
                    "rewritten_question": rewritten_question,
                    "rewritten_choices": rewritten_choices,
                    "subject": item.get("subject", ""),
                    "answer": item["answer"]
                })

        except Exception as e:
            print(f" Error with question:\n{question}\n{e}")
            

    # Save output
    with open("rewritten_mmlu_questions.json", "w") as f:
        json.dump(rewritten_data, f, indent=2)

    print(" Saved rewritten questions to 'rewritten_mmlu_questions.json'")

if __name__ == "__main__":
    main()
