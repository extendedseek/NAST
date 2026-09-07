# Contributing

1. Open an issue describing the methodological or implementation change.
2. Add or update tests for every behavior change.
3. Run `make test` and `make lint`.
4. Keep paper-faithful behavior separate from exploratory alternatives in configuration.
5. Never add unverified paper numbers, downloaded model weights, datasets, access tokens, or personally identifying data.

Bug reports should include the resolved YAML, environment metadata, smallest failing example, traceback, and whether the failure occurs in `loss_only` or `detach` mode.
