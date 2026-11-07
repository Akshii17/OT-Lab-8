# Failure log (AI-generated code that was wrong or misleading)

Cases found while the code was generated and run. Add your own as you test; the assignment asks for at least three and you must be able to explain them.

1. **Track H (DBMS): index size measured as 0 KB.** First version used `PRAGMA page_count` after dropping and re-creating indexes; SQLite reuses freed pages, so sizes came out as 0 and the greedy method crashed with division by zero. Fix: use `page_count - freelist_count`. Lesson: check measured inputs before trusting an optimizer's output.
2. **Track J: ILP hit the time limit.** The first instance (400 tests x 1500 lines) was too big for the exact solver within 60 s, so "ILP" was not actually optimal. Fix: smaller instance (150 x 500) that solves exactly. Lesson: an exact method labelled optimal must actually finish.
3. **Track D: GradNorm weight update contained a dead term (`*0`).** The generated formula had a leftover multiplied-by-zero expression. Fix: removed it. Lesson: read generated formulas line by line.
4. **Track G: xv6 patch was not compiled or booted.** No RISC-V toolchain/QEMU was available in the generation environment. The patch is untested. You must build it, boot it, and run `usertests` before submitting.
5. **Track H timing noise:** the ILP run reported a 1.4x speedup while choosing the same index as the DP run (2.1x). Same index, different timing: benchmark noise, not a method difference.
