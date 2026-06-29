import { Link, Outlet } from "@tanstack/react-router";

export function RootLayout() {
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
        </nav>
        <div className="sidebar-footer">tuner stats</div>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
