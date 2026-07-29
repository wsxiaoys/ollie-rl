// Adapted from open-thoughts/CodeContests task code_contests-0000.

const solutionPath = Bun.env.SOLUTION_PATH ?? "/workspace/solution.ts";
const testDataPath = Bun.env.TEST_DATA_PATH ?? "/tests/test_data.json";

interface TestData {
  inputs: string[];
  outputs: string[];
}

function fail(message: string): never {
  throw new Error(message);
}

if (!(await Bun.file(solutionPath).exists())) {
  fail(`solution not found: ${solutionPath}`);
}

const testData = (await Bun.file(testDataPath).json()) as TestData;
if (testData.inputs.length !== testData.outputs.length) {
  fail("test input/output counts do not match");
}

for (const [index, inputData] of testData.inputs.entries()) {
  const result = Bun.spawnSync({
    cmd: ["bun", solutionPath],
    stdin: Buffer.from(inputData),
    stdout: "pipe",
    stderr: "pipe",
    timeout: 10_000,
  });

  if (result.exitCode !== 0) {
    const stderr = new TextDecoder().decode(result.stderr);
    fail(`test ${index + 1} exited with code ${result.exitCode}: ${stderr}`);
  }

  const expectedLines = testData.outputs[index].trim().split("\n");
  const actualLines = new TextDecoder()
    .decode(result.stdout)
    .trim()
    .split("\n");
  if (JSON.stringify(actualLines) !== JSON.stringify(expectedLines)) {
    fail(
      `test ${index + 1} failed: expected ${JSON.stringify(expectedLines)}, ` +
        `got ${JSON.stringify(actualLines)}`,
    );
  }

  console.log(`test ${index + 1} passed`);
}
