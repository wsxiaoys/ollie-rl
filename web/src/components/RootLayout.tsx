import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { Link, Outlet, useNavigate, useRouterState } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { tunersQuery } from "../api/queries";

function getInitialTheme(): "dark" | "light" {
  if (typeof document !== "undefined") {
    const stored = document.documentElement.getAttribute("data-theme");
    if (stored === "light" || stored === "dark") return stored;
  }
  return "dark";
}

export function RootLayout() {
  const [theme, setTheme] = useState<"dark" | "light">(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const navigate = useNavigate();
  const { pathname, search } = useRouterState({ select: (s) => s.location });
  const { data } = useQuery(tunersQuery);
  const tuners = data?.tuners ?? [];

  // Resolve the active tuner from the current location — either the
  // `/tuners/$tunerId` path or the `?tuner=` search on the runs page — and fall
  // back to the first available tuner so the sidebar is always populated.
  const routeTunerId = useMemo(() => {
    const m = pathname.match(/^\/tuners\/([^/]+)/);
    if (m) return m[1];
    return (search as { tuner?: string }).tuner;
  }, [pathname, search]);
  const activeTunerId = routeTunerId ?? tuners[0]?.tuner_id;

  // Which section the current route belongs to. Run detail pages live under
  // `/tuners/$tunerId/runs/...`, so they count as "Runs" rather than "General".
  const onRuns =
    pathname.startsWith("/runs") || /^\/tuners\/[^/]+\/runs/.test(pathname);
  const onGeneral = /^\/tuners\/[^/]+$/.test(pathname);

  // Switching tuner keeps you in the same section (General vs Runs).
  const onSelectTuner = (e: ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    if (!id) return;
    if (onRuns) {
      navigate({ to: "/runs", search: { tuner: id } });
    } else {
      navigate({ to: "/tuners/$tunerId", params: { tunerId: id } });
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">🛹 ollie-rl</div>

        <div className="tuner-picker">
          <label htmlFor="sidebar-tuner">Tuner</label>
          <select
            id="sidebar-tuner"
            value={activeTunerId ?? ""}
            onChange={onSelectTuner}
            disabled={tuners.length === 0}
          >
            {tuners.length === 0 && <option value="">No tuners</option>}
            {tuners.map((t) => (
              <option key={t.tuner_id} value={t.tuner_id}>
                {t.name}
              </option>
            ))}
          </select>
        </div>

        <nav className="nav">
          {activeTunerId ? (
            <>
              <Link
                to="/tuners/$tunerId"
                params={{ tunerId: activeTunerId }}
                className={"nav-link" + (onGeneral ? " nav-link--active" : "")}
              >
                General
              </Link>
              <Link
                to="/runs"
                search={{ tuner: activeTunerId }}
                className={"nav-link" + (onRuns ? " nav-link--active" : "")}
              >
                Runs
              </Link>
            </>
          ) : (
            <>
              <span className="nav-link nav-link--disabled">General</span>
              <span className="nav-link nav-link--disabled">Runs</span>
            </>
          )}
        </nav>

        <div className="sidebar-footer">
          <button
            type="button"
            className="theme-toggle"
            aria-label="Toggle theme"
            title={theme === "dark" ? "Switch to light" : "Switch to dark"}
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
