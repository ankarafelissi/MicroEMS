"""Dashboard CSS kept in a Python module so Streamlit reloads style edits."""

STYLES = """
:root {
    color-scheme: dark;
    --canvas: #0b0d10;
    --panel: #111419;
    --line: #252b33;
    --muted: #8e99a8;
    --accent: #c2ef80;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp { background: var(--canvas); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, [data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] {
    max-width: 1600px;
    padding: 2.2rem 3rem 3rem;
}
[data-testid="stVerticalBlock"] { gap: 1rem; }
[data-testid="stSidebar"] {
    background: #101216;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebarContent"] { padding-top: 1.3rem; }
[data-testid="stSidebarUserContent"] { padding: 0 1.5rem 2rem; }
.sidebar-heading { margin: 0 0 1.6rem; }
.sidebar-heading h2 {
    font-size: 1.3rem;
    font-weight: 600;
    margin: .55rem 0 0;
    letter-spacing: .03em;
}
.eyebrow {
    color: var(--muted);
    font-size: .65rem;
    font-weight: 600;
    letter-spacing: .18em;
    margin: 0;
}
.masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.3rem;
    border-bottom: 1px solid var(--line);
}
.wordmark {
    display: flex;
    align-items: center;
    gap: .65rem;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: .08em;
}
.brand-symbol {
    display: inline-flex;
    color: var(--accent);
    background: #1c2518;
    width: 33px;
    height: 33px;
    align-items: center;
    justify-content: center;
    border: 1px solid #35472a;
    border-radius: 9px;
}
.brand-symbol svg { width: 23px; height: 23px; }
.brand-divider { height: 15px; width: 1px; background: #343b45; margin: 0 .45rem; }
.brand-label { font-size: .6rem; color: var(--muted); letter-spacing: .15em; font-weight: 500; }
.system-scope { color: #b3bcc8; font-size: .72rem; letter-spacing: .12em; }
.system-scope span { color: #414a57; margin: 0 .5rem; }
.masthead-actions { display: flex; align-items: center; gap: 1.6rem; }
.guide-link { display: inline-flex; align-items: center; gap: .65rem; padding: .45rem .7rem; border: 1px solid #35442b; border-radius: 6px; background: #182014; color: #cee6b9 !important; font-size: .72rem; text-decoration: none !important; white-space: nowrap; }
.guide-link:hover { background: #26331e; border-color: #64764f; }
.page-heading {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 2rem 0 1rem;
}
.page-heading h1 {
    color: #f3f5f7;
    font-size: 2.1rem;
    font-weight: 600;
    letter-spacing: -.04em;
    line-height: 1.3;
    margin: .6rem 0 0;
    padding: 0;
}
.workspace-tag {
    color: #a5aebb;
    font-size: .7rem;
    border: 1px solid #2c333c;
    border-radius: 6px;
    padding: .45rem .7rem;
    margin-bottom: .2rem;
}
.control-heading { display: flex; align-items: center; justify-content: space-between; color: #b6bfcc; font-size: .8rem; margin-top: .4rem; }
.plan-id { color: #838f9f; font-family: "Consolas", monospace; font-size: .67rem; letter-spacing: .08em; }
h3 { font-size: 1rem !important; font-weight: 600 !important; letter-spacing: .02em; padding: 1.2rem 0 .15rem !important; }
[data-testid="stCaptionContainer"] { color: var(--muted); font-size: .72rem; line-height: 1.65; }
[data-testid="stWidgetLabel"] p { font-size: .78rem; color: #b7c0cc; }
[data-testid="stForm"] { border: 0; padding: 0; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .85rem; }
[data-testid="stSidebar"] [data-testid="stExpander"] { margin: .4rem 0; }
[data-testid="stExpander"] { background: #12161b; border-color: var(--line); border-radius: 8px; }
[data-testid="stExpander"] summary p { font-size: .78rem; }
[data-baseweb="input"], [data-baseweb="select"] > div {
    background: #181c22;
    border-color: #303741;
    border-radius: 7px;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div { border-color: var(--accent); }
[data-testid="stNumberInput"] input, [data-baseweb="select"] { font-size: .8rem; }
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
    min-height: 2.65rem;
    border-radius: 7px;
    border-color: #303741;
    font-weight: 500;
    transition: background .15s ease, border-color .15s ease;
}
.stButton button p, .stDownloadButton button p, [data-testid="stFormSubmitButton"] button p { font-size: .78rem; }
button[kind="primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    color: #14200c;
    background: var(--accent);
    border-color: var(--accent);
}
button[kind="primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: #14200c; font-weight: 650; }
button[kind="primary"]:hover:not(:disabled), [data-testid="stBaseButton-primaryFormSubmit"]:hover { background: #d0f59f; border-color: #d0f59f; color: #14200c; }
.stButton button[kind="secondary"] { background: #15191f; color: #c7d0dc; }
.stButton button[kind="secondary"]:hover:not(:disabled) { background: #20262d; border-color: #556348; color: #e4f3d3; }
.stButton button:disabled { opacity: .38; }
.stButton button[kind="primary"]:disabled { background: #20291b; border-color: #35422b; opacity: .65; }
.stButton button[kind="primary"]:disabled p { color: #a6b39a; }
[data-testid="stMetric"] {
    background: linear-gradient(130deg, #181d22, #12161b 80%);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.25rem 1.3rem;
    min-height: 120px;
}
[data-testid="stMetricLabel"] { color: #a4aebd; }
[data-testid="stMetricLabel"] p { font-size: .75rem; }
[data-testid="stMetricValue"] { color: #f1f4f8; font-family: "Inter", "Segoe UI", sans-serif; font-weight: 500; letter-spacing: -.04em; font-size: 1.85rem; }
[data-testid="stMetric"]::before { content: ""; display: block; width: 18px; height: 2px; background: var(--accent); margin-bottom: .9rem; }
.run-status { display: flex; align-items: center; justify-content: flex-end; gap: .5rem; white-space: nowrap; font-size: .7rem; color: #c1cbd6; padding-top: 1.4rem; }
.run-status i { width: 6px; height: 6px; flex-shrink: 0; border-radius: 50%; background: var(--accent); }
.run-status span { color: var(--muted); font-family: "Consolas", monospace; margin-left: .4rem; }
.run-status.blocked i { background: #ed8f87; }
.run-status.paused i { background: #e9b96b; }
[data-testid="stProgress"] { margin-top: -.6rem; }
[data-testid="stProgress"] [role="progressbar"] { height: 3px; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.7rem; border-bottom: 1px solid var(--line); margin-bottom: 1.25rem; }
[data-testid="stTabs"] [data-baseweb="tab"] { padding: .8rem 0; height: auto; background: transparent; color: #929eaf; }
[data-testid="stTabs"] [data-baseweb="tab"] p { font-size: .8rem; }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--accent); }
[class*="st-key-chart_"] { background: var(--panel); border-color: var(--line) !important; border-radius: 10px; padding: 1.2rem 1.1rem .2rem; }
[class*="st-key-chart_"] [data-testid="stVerticalBlock"] { gap: .2rem; }
[class*="st-key-chart_"] [data-testid="stMarkdownContainer"] > p { font-size: .86rem; margin: 0; }
[class*="st-key-chart_"] strong { font-weight: 550; color: #dde4ec; }
[class*="st-key-chart_"] [data-testid="stCaptionContainer"] p { font-size: .65rem; }
[data-testid="stTable"], [data-testid="stDataFrame"] { border-color: var(--line); border-radius: 8px; overflow: hidden; }
[data-testid="stTable"] th { background: #191e25; color: #b4bfce; font-size: .75rem; font-weight: 500; }
[data-testid="stTable"] td { font-size: .78rem; border-color: var(--line); }
[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stAlert"] p { font-size: .8rem; }
hr { border-color: var(--line); margin: 1.2rem 0; }
.empty-workspace { text-align: center; padding: 3.7rem 1rem 3.3rem; border: 1px solid var(--line); border-radius: 12px; background: radial-gradient(ellipse at 50% 10%, #1b241980, transparent 62%), var(--panel); }
.empty-orbit { display: flex; align-items: center; justify-content: center; width: 100px; height: 100px; margin: 0 auto 1.7rem; color: var(--accent); border: 1px solid #35412d; border-radius: 50%; background: #202b1950; box-shadow: 0 0 0 13px #c2ef8005, 0 0 0 27px #c2ef8003; }
.empty-orbit svg { width: 68px; height: 68px; }
.empty-workspace h2 { font-size: 1.6rem; font-weight: 500; letter-spacing: -.02em; padding: 1rem 0 .4rem; }
.empty-workspace > p:not(.eyebrow) { color: var(--muted); font-size: .8rem; }
.empty-specs { display: flex; justify-content: center; gap: 3rem; margin-top: 2.5rem; }
.empty-specs span { display: flex; flex-direction: column; gap: .5rem; color: #919eae; font-size: .7rem; }
.empty-specs b { color: #dbe2ec; font-size: 1.5rem; font-weight: 400; font-variant-numeric: tabular-nums; }
:focus-visible { outline-color: var(--accent); }
@media (max-width: 1100px) {
    [data-testid="stMainBlockContainer"] { padding: 2rem 1.5rem; }
    .brand-label { display: none; }
    .run-status { flex-wrap: wrap; white-space: normal; }
    [data-testid="stMetric"] { padding: 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; }
    .masthead { padding-top: .9rem; }
    .system-scope { font-size: .65rem; letter-spacing: 0; }
    .system-scope span { margin: 0 .2rem; }
    .masthead-actions { gap: .6rem; }
    .masthead-actions .system-scope { display: none; }
    .brand-divider, .workspace-tag { display: none; }
    .page-heading { padding: 1.5rem 0 .5rem; }
    .page-heading h1 { font-size: 1.7rem; }
    .run-status { justify-content: flex-start; padding: 0; }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"] {
        width: 100%;
        min-width: 0;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.2rem; }
    .empty-workspace { padding: 2.7rem .8rem; }
    .empty-workspace h2 { font-size: 1.25rem; }
    .empty-specs { gap: 2rem; }
}
"""
