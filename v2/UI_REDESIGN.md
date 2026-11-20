# DataCorn workspace redesign

The UI now uses a dedicated navigation rail and a top dataset toolbar. Dataset import and demo loading are under Manage data. The sidebar contains navigation and account controls only.

- Overview: actual dataset metrics, recent investigations, record/field tabs, data readiness and model entry points.
- Investigate: question and plan editor alongside computed evidence, with progress steps and empty/stale-result states.
- Data health: summary metrics and separate quality, column-profile and data-preview tabs.
- Models: training configuration separated from results, scenarios and saved jobs.
- Evidence: decision brief, export contents and PDF/Excel/JSON downloads.
- History: question search, saved-investigation reopening and explicit dataset-removal confirmation.

Icons accompany text labels. Decorative vector icons are inline assets; no remote imagery is required. Styles include keyboard focus, reduced-motion support and narrow-screen stacking. Native Streamlit controls retain their labels and keyboard behavior.

## Local launch

Run `start-local.cmd` from this project folder. It prefers `.venv-v2` and opens the server at `http://127.0.0.1:8503`. Install the pinned requirements with `setup-local.cmd` if that environment is incomplete.

## Verification and release status

Python syntax and theme configuration checks passed. The UI regression suite was updated for button navigation and a new all-page/dataset-switch test was added.

Backend regression result: **53 passed**, including CSV/XLSX import workflows, investigations, statistical review, model persistence, explanations and evidence exports. Tests ran with the project virtual environment, with pytest loaded from the system installation; four optional-package/version warnings were reported. This result does not validate the rendered UI.

Live UI verification remains blocked in the editing session: Streamlit AppTest stalls creating the Windows asyncio event loop, in `socket._fallback_socketpair` / `accept`, before the application renders. The diagnostic run stopped after 40 seconds. No browser surface was available for screenshots or responsive-layout inspection. These checks must pass before production sign-off; the redesign alone is not a production-readiness certification.
