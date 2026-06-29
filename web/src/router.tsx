import {
  createRootRoute,
  createRoute,
  createRouter,
  redirect,
} from "@tanstack/react-router";
import { RootLayout } from "./components/RootLayout";
import { TunerListPage } from "./pages/TunerListPage";
import { TunerDetailPage } from "./pages/TunerDetailPage";

const rootRoute = createRootRoute({
  component: RootLayout,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: () => {
    throw redirect({ to: "/tuners" });
  },
});

const tunersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tuners",
  component: TunerListPage,
});

const tunerDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tuners/$tunerId",
  component: TunerDetailPage,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  tunersRoute,
  tunerDetailRoute,
]);

export const router = createRouter({
  routeTree,
  basepath: "/app",
  defaultPreload: "intent",
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
