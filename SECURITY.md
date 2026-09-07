# Security policy

Do not open a public issue for a suspected vulnerability involving arbitrary code execution, dependency confusion, credential exposure, or unsafe model artifacts. Contact the repository owner privately and include a minimal reproduction without real credentials.

Only load models/datasets from trusted owners and immutable revisions. Leave `trust_remote_code: false` unless the code has been reviewed. Never commit Hugging Face, Weights & Biases, or other service tokens.
