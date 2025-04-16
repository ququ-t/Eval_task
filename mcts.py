import math
import random
import json
import os
from typing import List
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(api_key="sk-4d5824bb72d54528bd45943def4688db", base_url="https://api.deepseek.com")

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) >= 2

    def best_child(self, c_param):
        return max(
            self.children,
            key=lambda child: (
                (child.reward / child.visits) if child.visits else float("inf")
            )
            + c_param * math.sqrt(
                math.log(self.visits) / (child.visits if child.visits else 1)
            )
        )

    def update(self, reward):
        self.visits += 1
        self.reward += reward

def selection(node, c_param):
    current = node
    while current.children:
        if not current.is_fully_expanded():
            break
        current = current.best_child(c_param)
    return current

def expansion(node):
    new_state = dict(node.state)
    chain = new_state.get("chain", [])
    new_state["chain"] = chain + [f"Step {len(chain) + 1}"]
    child = Node(new_state, parent=node)
    node.children.append(child)
    return child

def deepseek_generate(state):
    prompt = f"Question: {state['question']}\n"
    if state.get("chain"):
        prompt += "Current reasoning:\n" + "\n".join(state["chain"]) + "\n"
    prompt += (
        "Continue the reasoning and provide the final answer. "
        "Use the delimiter 'Final Answer:' to separate the reasoning from the answer."
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that reasons step by step."},
        {"role": "user", "content": prompt},
    ]
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=True
    )
    
    reasoning_content = ""
    content = ""

    print("sreasoning_content")

    for chunk in response:
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
            reasoning_content += chunk.choices[0].delta.reasoning_content
        elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    
    full_content = reasoning_content + content
    
    if "Final Answer:" in full_content:
        reasoning_part, answer_part = full_content.split("Final Answer:", 1)
        chain = [line.strip() for line in reasoning_part.splitlines() if line.strip()]
        answer = answer_part.strip()
    else:
        chain = [line.strip() for line in full_content.splitlines() if line.strip()]
        answer = full_content
    
    return {"chain": chain, "answer": answer}

def get_outcome_reward(full_chain: List[str]) -> float:
    if not full_chain:
        return 0.0
    final_line = full_chain[-1].lower()
    return 1.0 if "4" in final_line else 0.0


def er_prm_reward(partial_chain: List[str], question: str, eta: float) -> float:
    exponentiated = []
    for i in range(3):
        # Make a copy of the state with partial_chain
        mock_state = {"question": question, "chain": partial_chain}
        gen = deepseek_generate(mock_state)
        print("iter_num is %d", i)
        full_chain = gen["chain"]
        final_reward = get_outcome_reward(full_chain)
        exponentiated.append(math.exp(eta * final_reward))

    partial_reward = (1.0 / eta) * math.log(sum(exponentiated) / len(exponentiated) + 1e-8) 
    return partial_reward


def rewards(state, eta):
    chain_so_far = state.get("chain", [])
    question = state["question"]

    partial_reward = er_prm_reward(chain_so_far, question, eta)
    return partial_reward


def backpropagation(node, reward):
    current = node
    while current:
        current.update(reward)
        current = current.parent

def mcts(root_state, num_iter, max_depth, c_param, eta):
    root = Node(root_state)
    for _ in range(num_iter):
        node = selection(root, c_param)
        if len(node.state.get("chain", [])) < max_depth:
            node = expansion(node)
        sim_reward = rewards(node.state, eta)
        backpropagation(node, sim_reward)
    return max(root.children, key=lambda n: n.visits).state if root.children else root.state

def main():
    # Load the MMLU dataset from Hugging Face.
    dataset = load_dataset("cais/mmlu", "all", split="test")
    samples = dataset.select(range(5))
    results = []
    correct = 0
    total = len(samples)
    counter = 0

    for sample in samples:
        counter += 1
        correct_choice = str(sample.get("answer", "")).strip().lower()
        state = {"question": sample["question"], "chain": []}
        best_state = mcts(state, num_iter=3, max_depth=5, c_param=1.4, eta=1.0)

        output = deepseek_generate(best_state)
        reasoner_answer = str(output.get("answer", "")).strip().lower()

        if correct_choice == reasoner_answer:
            correct += 1
        result = {
            "question": sample["question"],
            "correct_choice": sample.get("answer", ""),
            "reasoning": output.get("chain", []),
            "reasoner_answer": output.get("answer", "")
        }
        results.append(result)
        # Print each result in the terminal
        print("====Qustion %d====", counter)
        print("Question:")
        print(result["question"])
        print("Ground Truth Answer:")
        print(result["correct_choice"])
        print("Reasoning:")
        for line in result["reasoning"]:
            print(line)
        print("Reasoner Answer:")
        print(result["reasoner_answer"])
        print("-" * 50)

    accuracy = correct / total if total else 0
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    output_data = {"results": results, "accuracy": accuracy}
    with open("mmlu_questions_reasoning.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("\nFinal Output JSON:")
    print(json.dumps(output_data, indent=2))

if __name__ == "__main__":
    main()
