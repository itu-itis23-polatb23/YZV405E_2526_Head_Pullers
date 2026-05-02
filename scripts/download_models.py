"""
scripts/download_models.py
──────────────────────────
NOTE: This step is no longer required.

Earlier revisions of the pipeline lemmatized constraint tokens locally with
spaCy / Stanza models. The current pipeline asks the LLM to return lemmas
during the detection stage (see src/pipeline.py::detect_mwe), and
src/lemmatizer.py is a simple regex tokenizer. spaCy and Stanza are not
installed by requirements.txt.

This script is kept as a no-op so the README setup flow does not error.
"""

import sys


def main():
    print(
        "download_models.py: nothing to do.\n"
        "  The pipeline no longer uses spaCy or Stanza models.\n"
        "  Lemmas are produced by the LLM during MWE detection.\n"
        "  You can skip this step and run run_pipeline.py directly."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
