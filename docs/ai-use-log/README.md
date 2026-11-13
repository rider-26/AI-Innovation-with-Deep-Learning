# AI Use Logs

One file per member, named after your GitHub username (for example `dav_1ann.md`). Copy
`TEMPLATE.md` to start yours. The log is submitted at PR1, PR2 and the Final, and you are
questioned on two or three entries live.

What the guidebook says (Section 7 and Annex B):

- Update it as you work, not at the end. Log anything that produced a block of code, a design
  decision, or a resource you then used. Autocomplete doesn't need logging.
- **The "Why" column is the one that is marked.** Write a specific check, a specific finding and a
  specific action. "Accepted, it worked" earns nothing.
- The log must agree with the commit history. A log that contradicts it is treated as misconduct.
- Commit bulk AI-generated code separately from your own edits, and put the commit hash in the log.

A good entry, from the guidebook:

| Date | Tool | What you asked for | Accepted / changed / rejected | Why: what you checked and what you found |
|---|---|---|---|---|
| 14 Feb | Copilot | Suggest an evaluation block for the triage classifier | Rejected | Computed accuracy on the training split. Rewrote to evaluate on the held-out test set: the original number was meaningless |
