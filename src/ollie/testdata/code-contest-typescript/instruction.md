# Balanced Brackets — TypeScript with Bun

Vipul needs help checking whether strings of parentheses are balanced.

## Input

The first line contains an integer `T`, the number of test cases. Each of the
next `T` lines contains a non-empty string made of `(` and `)` characters.

Constraints:

- `1 <= T <= 10`
- `1 <= len(S) <= 60`

## Output

For each test case, print `YES` when the parentheses are balanced and `NO`
otherwise.

A string is balanced when every opening parenthesis has a matching closing
parenthesis and no prefix contains more closing than opening parentheses.

## Example

Input:

```text
3
((()))
(())()
()(()
```

Output:

```text
YES
YES
NO
```

Create a TypeScript solution at `solution.ts` in the current workspace
(`/workspace/solution.ts`). Before finishing, run it with Bun against the
example input and confirm that it produces the expected output.

The final solution must be executable with `bun /workspace/solution.ts`, read
from standard input, and write only the requested answers to standard output.
Do not submit a JavaScript or Python solution.
