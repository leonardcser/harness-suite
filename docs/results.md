# Reading the results

Every case comes from the [spec](../SPEC.md).

## Outcomes

| Outcome       | Meaning                                                          |
| ------------- | ---------------------------------------------------------------- |
| `pass`        | The agent did the right thing                                    |
| `fail`        | It broke a required rule or a claim it made                      |
| `gap`         | A missing safeguard, not a failure                               |
| `unsupported` | The agent says it lacks the feature                              |
| `label`       | A short note on a choice that reasonable agents make differently |
| `skip`        | Not measured: the run never reached what the case judges         |

## What counts

- A fail or gap needs harm the suite can see: a hang, a silent failure, a
  request the API rejects.
- How long to retry on a dead network is a label, or passes either way.
- Unsupported comes only from the agent itself: its recipe, an `[unsupported]`
  reason, or a capability it does not declare.
- `request.valid` is a required check that every case feeds. A request the real
  API would reject fails it, unless the case provokes that rule on purpose.

## Repeatable

The suite scripts every step, so the same agent version gives the same verdicts.
If the plain tool call (`tools.round-trip` `text`) fails, the other tool cases
of that agent are not measured instead of failing again.
