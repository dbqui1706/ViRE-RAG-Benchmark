---
description: "Load Antigravity knowledge and apply it to the current ViRE workspace task"
name: "Use Antigravity Knowledge"
argument-hint: "Describe the task to run using your organized knowledge"
agent: "agent"
---
You are a coding assistant working in the current workspace.

Goal:
- Access the knowledge base at path: C:\Users\PC\.gemini\antigravity\knowledge
- Find, read, and synthesize knowledge directly relevant to the user request
- Apply the knowledge to this project code/docs instead of giving generic advice

Required process:
1. Verify access to the knowledge path and list a short folder structure summary.
2. Identify the most relevant files/sections for the user request.
3. Extract actionable insights (code conventions, workflow, commands, patterns, decision rationale).
4. Execute the task in the current workspace using the extracted knowledge.
5. Return output with 2 sections:
   - "Applied Knowledge": what was used from the knowledge base.
   - "Project Changes/Output": what was changed or produced.

Quality constraints:
- If the path is inaccessible or permission is denied, report the exact reason and propose a practical fallback.
- Prefer the most recent and specific information; avoid guessing.
- If sources conflict, state which source is prioritized and why.
- Respond in Vietnamese unless the user explicitly asks for another language.

User input:
- Use the invocation argument as the main task to execute.
