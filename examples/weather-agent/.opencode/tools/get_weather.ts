import { tool } from "@opencode-ai/plugin"
import path from "path"
import { readFileSync } from "fs"

// A simple static "weather provider" that reads its data from a local JSON
// file. The tool is deterministic in terms of the underlying temperature,
// but intentionally *randomises* the unit it reports in:
//
//   - ~50% of calls: returns Fahrenheit (the value as stored).
//   - ~50% of calls: returns Celsius (converted from the stored value).
//
// This forces the agent to read the `unit` field and, when it sees °C,
// convert back to Fahrenheit before answering — which is the actual
// behaviour our reward function is grading.

export default tool({
  description:
    "Get the current weather for a city. Returns the temperature in either Fahrenheit or Celsius (see the `unit` field of the response) along with a short condition string.",
  args: {
    city: tool.schema
      .string()
      .describe("The name of the city, e.g. 'San Francisco'"),
  },
  async execute(args, context) {
    const dataPath = path.join(
      context.worktree,
      "examples/weather-agent/data/cities.json",
    )
    const raw = readFileSync(dataPath, "utf-8")
    const db = JSON.parse(raw) as Record<
      string,
      { fahrenheit: number; condition: string }
    >

    const key = Object.keys(db).find(
      (k) => k.toLowerCase() === args.city.toLowerCase(),
    )
    if (!key) {
      return JSON.stringify({
        error: `No weather data for city '${args.city}'.`,
        known_cities_sample: Object.keys(db).slice(0, 5),
      })
    }
    const entry = db[key]

    // Randomly pick the unit the tool reports in. The temperature is the
    // same physical quantity either way; we just present it differently.
    const reportInCelsius = Math.random() < 0.5
    const temperature = reportInCelsius
      ? Math.round(((entry.fahrenheit - 32) * 5) / 9)
      : entry.fahrenheit
    const unit = reportInCelsius ? "°C" : "°F"

    return JSON.stringify({
      city: key,
      temperature,
      unit,
      condition: entry.condition,
    })
  },
})
