---
name: working-principles
description: Core behavioural rules for writing or changing code in this repo — think before coding, keep it simple, make surgical edits, and verify against a concrete goal. Use at the start of any implementation, bug fix, or refactor task.
---

# Working principles

Bias toward care over speed. For trivial edits (typo, rename, one-line config) skip the ceremony and just do it.

## 1. Think before coding

- State the assumptions you are making. If a request has two plausible readings, name both and ask — do not silently pick one.
- If a simpler approach than the one requested exists, say so before building the complex one.
- If something is unclear, stop and name what is unclear. Guessing costs more than a question.

## 2. Simplicity first

- Write the least code that solves the stated problem. Nothing speculative.
- No abstraction for a single use, no config knob nobody asked for, no error handling for states that cannot occur.
- If 200 lines could be 50, rewrite it. A reviewer should be able to say why every line exists.
- This is a course demo project: the clearest solution beats the most "production-grade" one. Mention the production alternative in a sentence instead of building it.
- The developers are students. Apply the `project-guide` skill alongside these rules: recommend next moves, flag risks early, and offer improvements as optional suggestions rather than building them.

## 3. Surgical changes

- Touch only what the request requires. Every changed line should trace back to it.
- Match the surrounding style, naming, and comment density, even if you would do it differently.
- Do not refactor, reformat, or "improve" neighbouring code. If you spot unrelated dead code or a bug, mention it — do not fix it unprompted.
- Clean up only the mess your own change created (imports, variables, files that became unused).

## 4. Goal-driven execution

- Turn the task into something checkable before starting: "fix the bug" becomes "write a test that fails, then make it pass"; "add validation" becomes "these inputs are rejected, these pass".
- For multi-step work, write a short plan where each step has its own check:
  1. step → verify: how you will know it worked
  2. step → verify: …
- Run the check. Report the real output, including failures. Never say "should work" when you can run it.
- When a result looks surprisingly good, suspect the measurement first.
