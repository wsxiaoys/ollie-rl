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
import { DatumsPage } from "./pages/DatumsPage";
import { EvalPage } from "./pages/EvalPage";
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
  ): { tuner?: string; datum?: string; kind?: "train" | "eval" } => ({
    tuner: typeof search.tuner === "string" ? search.tuner : undefined,
    datum: typeof search.datum === "string" ? search.datum : undefined,
    kind: search.kind === "eval" ? "eval" : undefined,
  }),
  component: RunListPage,
});

const datumsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/datums",
  validateSearch: (
    search: Record<string, unknown>,
  ): { tuner?: string; datum?: string } => ({
    tuner: typeof search.tuner === "string" ? search.tuner : undefined,
    datum: typeof search.datum === "string" ? search.datum : undefined,
  }),
  component: DatumsPage,
});

const evalRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tuners/$tunerId/eval",
  component: EvalPage,
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
  datumsRoute,
  evalRoute,
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
