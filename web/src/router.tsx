import {
  createRootRoute,
  createRoute,
  createRouter,
  redirect,
} from "@tanstack/react-router";
import { RootLayout } from "./components/RootLayout";
import { TunerListPage } from "./pages/TunerListPage";
import { TunerDetailPage } from "./pages/TunerDetailPage";
import { RunListPage } from "./pages/RunListPage";
import { DataPage } from "./pages/DataPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { CompletionDetailPage } from "./pages/CompletionDetailPage";

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

const runListRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs",
  validateSearch: (
    search: Record<string, unknown>,
  ): { tuner?: string; datum?: string } => ({
    tuner: typeof search.tuner === "string" ? search.tuner : undefined,
    datum: typeof search.datum === "string" ? search.datum : undefined,
  }),
  component: RunListPage,
});

const dataRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/data",
  validateSearch: (
    search: Record<string, unknown>,
  ): { tuner?: string; datum?: string } => ({
    tuner: typeof search.tuner === "string" ? search.tuner : undefined,
    datum: typeof search.datum === "string" ? search.datum : undefined,
  }),
  component: DataPage,
});

const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tuners/$tunerId/runs/$runId",
  component: RunDetailPage,
});

const completionDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tuners/$tunerId/runs/$runId/completions/$completionId",
  component: CompletionDetailPage,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  tunersRoute,
  tunerDetailRoute,
  runListRoute,
  dataRoute,
  runDetailRoute,
  completionDetailRoute,
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
