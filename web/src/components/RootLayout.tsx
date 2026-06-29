import { useEffect, useState } from "react";
import { Link, Outlet } from "@tanstack/react-router";

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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">🛹 ollie-rl</div>
        <nav className="nav">
          <Link
            to="/tuners"
            className="nav-link"
            activeProps={{ className: "nav-link nav-link--active" }}
          >
            Tuners
          </Link>
          <Link
            to="/runs"
            className="nav-link"
            activeProps={{ className: "nav-link nav-link--active" }}
          >
            Runs
          </Link>
        </nav>
        <div className="sidebar-footer">
          <span>tuner stats</span>
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
