# Verified fixes in the working copy

- Questions naming revenue or sales now choose that outcome ahead of an unrelated churn column.
- Column-name matching recognizes spaces and punctuation, alongside underscores.
- Empty and one-column schemas produce an actionable planning error.
- Editing a plan hides the old finding and challenge controls until the analysis is rerun; the old investigation remains in History.
- Rejected training submissions preserve the currently opened model, scenario and prepared exports.
- Scenario class indices are validated instead of silently clamped to the last class.
- A missing skops package no longer crashes the investigation workspace at import time. Model saving/loading reports the missing dependency explicitly; training is disabled until it is installed.

Start this working copy with start-local.cmd from the repository root. It opens the current v2 application in local mode on port 8503. The older FastAPI/Streamlit application on port 8501 is a separate entrypoint.

Observed checks: 46 tests passed, 4 deselected, 2 warnings. Python compilation passed.
The four deselected tests require the unavailable skops serializer: linear and forest artifact roundtrips, background training publication, and artifact type checking. The full UI suite stalled in this restricted execution session and was stopped; the UI changes are not claimed as live-browser verified. The new UI regression test is included for a normal installed environment.

All changes are in the shared working copy. The source originally supplied on G: was not overwritten. A candidate moved source was found under G:/Projects/DataCorn/AutoInsight-2.1, but its recreated environment did not contain all pinned dependencies.
