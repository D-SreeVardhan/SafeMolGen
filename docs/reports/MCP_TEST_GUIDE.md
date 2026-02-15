# MCP Browser Automation – Setup & Test Guide

**App URL:** http://localhost:5173 (React frontend; backend at http://localhost:8000)  
**Purpose:** Enable Cursor to drive the SafeMolGen-DrugOracle UI via MCP and run automated browser tests.

---

## 1. MCP configuration (already in project)

The project includes a **project-level** MCP config so the Browser MCP server is available when you open this repo in Cursor:

- **File:** `.cursor/mcp.json` (at workspace root: `MiniProject/.cursor/mcp.json`)
- **Server:** `browsermcp` via `npx -y @browsermcp/mcp@latest`

If Cursor does not pick it up automatically:

1. Open **Cursor Settings** → **Tools** → **MCP**.
2. Add or confirm a server with:
   - **Name:** `browsermcp`
   - **Command:** `npx`
   - **Args:** `-y`, `@browsermcp/mcp@latest`
3. Click **Refresh** next to the server.

**Prerequisites:**

- **Node.js** installed (for `npx`).
- **Browser MCP extension** installed in the browser you want to automate (see [Set up extension](https://docs.browsermcp.io/setup-extension)).

---

## 2. Start the app before testing

From the `SafeMolGen-DrugOracle` directory:

1. **Backend:** `python scripts/run_app.py` or `uvicorn backend.main:app --port 8000`
2. **Frontend:** `cd frontend && npm install && npm run dev`

Default frontend URL: **http://localhost:5173** (proxies /api to backend). Leave both running while you run MCP tests.

---

## 3. Test flow (for agent or human)

When Browser MCP tools are available (e.g. `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`), use this flow.

### 3.1 Lock/unlock

1. **Navigate:** `browser_navigate` → `http://localhost:5173`
2. **Lock:** `browser_lock` (required before interactions)
3. After all steps: **Unlock:** `browser_unlock`

### 3.2 Generate

1. **Snapshot** to get the page structure and element refs.
2. Select **Generate** in the sidebar (click the corresponding element).
3. Optionally set: Safety threshold (e.g. 0.02), Max iterations (e.g. 3–5), "Use RL model".
4. **Click** the "Run generation" button.
5. **Wait** for results (short waits + snapshots until Best Molecule, Oracle dashboard, Optimization Journey, Recommendations appear).
6. **Snapshot** and verify: Best Molecule card, Oracle dashboard, Optimization Journey, Recommendations.

### 3.3 Analyze

1. Switch mode to **Analyze** (sidebar).
2. **Type** or **fill** SMILES in the input (e.g. `CC(=O)Oc1ccccc1C(=O)O`).
3. **Click** "Analyze".
4. **Snapshot** and verify: molecule card, Oracle dashboard, recommendations.

### 3.4 Compare

1. Switch mode to **Compare**.
2. Enter 2+ SMILES (one per line) in the text area.
3. **Click** "Compare".
4. **Snapshot** and verify: comparison table.

### 3.5 About

1. Switch mode to **About**.
2. **Snapshot** and verify overview content.

---

## 4. If Browser MCP tools are not available

- Confirm `.cursor/mcp.json` is at the workspace root and contains the `browsermcp` server.
- In Cursor: **Settings → Tools → MCP** → Refresh the server; restart Cursor if needed.
- Ensure Node.js is installed and the Browser MCP extension is installed in your browser.
- Manual UI checklist: see **docs/reports/APP_TEST_STATUS.md**.

---

## 5. Playwright E2E tests (no MCP required)

You can run browser tests from the command line without Cursor MCP:

```bash
# One-time setup
pip install -r requirements-e2e.txt
playwright install chromium   # required; if tests say "Executable doesn't exist", run this

# Start the app in another terminal: backend (python scripts/run_app.py) and frontend (cd frontend && npm run dev)

# Run E2E tests
python -m pytest tests/e2e_browser_test.py -v
# or
python tests/e2e_browser_test.py
```

Tests: app loads, About page, Analyze with SMILES. Override URL with `E2E_APP_URL` (default http://localhost:5173). If tests fail with "Executable doesn't exist", run `playwright install chromium` once.

---

## 6. Updating test status

After running the MCP-driven UI test or Playwright E2E tests, you can update **docs/reports/APP_TEST_STATUS.md**:

- Set the "UI (in-browser)" row to **PASS** and add a note: "UI tested via MCP (Generate, Analyze, Compare, About)."
- Or note any failing step (e.g. "Generate: button not found", "Compare: table empty").
