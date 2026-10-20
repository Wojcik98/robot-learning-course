# Concept guides

These guides explain the methods before you implement their parts. They assume that you know Python, arrays, a training loss, and gradient descent. Each guide starts with the problem, follows one example through the system, then locates the notebook exercises in that system.

Open a guide in Markdown Preview beside its notebook. Mermaid diagrams are in these Markdown files only. Notebook explanations contain the essential context and use inline SVG diagrams or tables. Each diagram also has a text explanation.

| Read this guide | Question it answers | Related lessons |
| --- | --- | --- |
| [From demonstrations to control](control-and-learning.md) | What does a policy learn, and how does a prediction move the robot? | 1–4, then all execution lessons |
| [ACT: predict an action chunk](act.md) | Why use chunks, a latent variable, and attention? | 2–9 |
| [Diffusion: generate actions from noise](diffusion-policy.md) | What is diffused, what conditions it, and how does sampling become control? | 10–15 |
| [SmolVLA: language context and action flow](smolvla.md) | How does a vision-language model connect to continuous robot actions? | 16–20 |
| [Data, adaptation, and evidence](data-and-evaluation.md) | What can a lower loss or a successful rollout establish? | 3–4, 9, 15, 20–22 |

Start with the control guide. Read the method guide when you start that method. Keep the lesson map open while you work through its parts. Return to the notebook for the exact tensor contract and the exercise.

A useful reading check is: **What enters this component, what leaves it, and why does the next component need that output?** If you cannot answer yet, read the example before you write code.

[Course instructions](../README.md) · [Notebook index](../index.ipynb)
